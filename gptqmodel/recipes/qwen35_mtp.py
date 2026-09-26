# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Calibrate Qwen dense/MoE MTP on HF or already quantized checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Qwen35MTPRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    provenance: dict


def replay_main_hidden_layerwise(owner, calibration, *, device):
    """Replay all samples through one main-model block at a time."""
    from time import perf_counter
    import torch
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from ..utils.looper_helpers import rehome_module_to_device

    language = owner.model.language_model
    language.eval()
    block_device = torch.device(device)
    records = []
    with torch.inference_mode():
        for sample in calibration:
            ids = sample["input_ids"].cpu()
            mask = sample.get("attention_mask", torch.ones_like(ids)).cpu()
            if ids.shape[0] != 1:
                raise ValueError("MTP calibration expects individual unpadded sequences")
            ids = ids[:, mask[0].bool()]
            if ids.shape[1] < 3:
                raise ValueError("MTP calibration needs at least three non-padding tokens")
            hidden = language.embed_tokens(ids)
            positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(4, 1, -1)
            text_positions = positions[0]
            rotary = language.rotary_emb(hidden, positions[1:])
            mask_kwargs = dict(config=language.config, inputs_embeds=hidden,
                               attention_mask=None, past_key_values=None,
                               position_ids=text_positions)
            masks = {"full_attention": create_causal_mask(**mask_kwargs),
                     "linear_attention": create_recurrent_attention_mask(**mask_kwargs)}
            records.append({"hidden": hidden.cpu(), "embedding": language.embed_tokens(ids[:, 1:]).cpu(),
                            "position_ids": text_positions, "position_embeddings": tuple(x.cpu() for x in rotary),
                            "masks": masks})

        for index, layer in enumerate(language.layers[:language.config.num_hidden_layers]):
            started = perf_counter()
            layer.to(block_device).eval()
            rehome_module_to_device(layer, block_device, move_parameters=True, move_buffers=True)
            try:
                kind = language.config.layer_types[index]
                for record in records:
                    attention_mask = record["masks"][kind]
                    if torch.is_tensor(attention_mask):
                        attention_mask = attention_mask.to(block_device)
                    output = layer(
                        record["hidden"].to(block_device),
                        position_embeddings=tuple(x.to(block_device) for x in record["position_embeddings"]),
                        attention_mask=attention_mask,
                        position_ids=record["position_ids"].to(block_device),
                        past_key_values=None, use_cache=False,
                    )
                    record["hidden"] = output.cpu()
            finally:
                layer.to("cpu")
                rehome_module_to_device(layer, torch.device("cpu"), move_parameters=True, move_buffers=True)
                if block_device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"[mtp] replayed main block {index + 1}/{language.config.num_hidden_layers} "
                  f"with {len(records)} samples in {perf_counter() - started:.1f}s", flush=True)

        for record in records:
            record["hidden"] = language.norm(record["hidden"]).cpu()
            record.pop("position_ids")
            record.pop("position_embeddings")
            record.pop("masks")
    return records


def collect_mtp_inputs(source, model, calibration, *, device, validation_nsamples, replay_device=None):
    """Replay the supplied main model, then align hidden[t] with token[t+1]."""
    import torch
    from .. import GPTQModel
    from ..utils.backend import BACKEND

    from ..quantization.lm_head import read_checkpoint_config
    replay_device = replay_device or device
    config = read_checkpoint_config(source)
    if config.get("quantization_config"):
        main = GPTQModel.load(str(source), device="cpu", backend=BACKEND.TORCH, torch_dtype=torch.float16)
        owner = main.model
    else:
        from transformers import AutoModelForImageTextToText
        owner = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float16, device_map={"": "cpu"})
        main = owner
    records = replay_main_hidden_layerwise(owner, calibration, device=replay_device)
    del main, owner
    gc.collect()
    model.to(device).eval()
    expert_inputs = []
    validation = []
    captured = []

    def capture(module, args):
        captured.append(args[0].detach().reshape(-1, model.config.hidden_size).cpu())

    handle = model.layers[0].mlp.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for index, record in enumerate(records):
                hidden = record["hidden"]
                embeddings = record["embedding"].to(device=device, dtype=model.fc.weight.dtype)
                mtp_hidden = hidden[:, :-1].to(device=device, dtype=model.fc.weight.dtype)
                reference = model(embeddings, mtp_hidden)
                if index < len(calibration) - validation_nsamples:
                    if model.config.model_type == "qwen3_5_text":
                        expert_inputs.append({"embedding": embeddings.cpu(), "hidden": mtp_hidden.cpu()})
                    else:
                        expert_inputs.extend(captured)
                else:
                    validation.append({"embedding": embeddings.cpu(), "hidden": mtp_hidden.cpu(),
                                       "reference": reference.cpu(), "main_next_hidden": hidden[:, 1:].cpu()})
                captured.clear()
                print(f"[mtp] collected sequence {index + 1}/{len(calibration)}, tokens={hidden.shape[1] - 1}", flush=True)
    finally:
        handle.remove()
        model.cpu()
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return expert_inputs, validation


def quantize_qwen35_mtp(*, model_dir, mtp, output_dir=None, calibration_jsonl=None,
                       calibration_dataset=None, calibration_text_key="text", nsamples=256,
                       seqlen=2048, device="cuda:0", replay_device=None, seed=42,
                       dry_run=False, calibration_cache=None):
    from ..quantization.qwen35_mtp import (
        MTPQuantizationConfig, Qwen35MTP, calibrate_mtp_experts, inspect_mtp_checkpoint, save_mtp_checkpoint,
        validate_mtp_payload, resolve_mtp_algorithm,
    )
    from .qwen35_head import _load_calibration, _validate_calibration_source

    options = MTPQuantizationConfig(**{"seed": seed, **mtp})
    if replay_device is not None and (not isinstance(replay_device, str) or not replay_device.strip()):
        raise ValueError("MTP replay_device must be a non-empty device string")
    replay_device = replay_device or device
    source = Path(model_dir).expanduser().resolve()
    # Full-recipe previews refer to a not-yet-created backbone checkpoint;
    # that caller already resolves the algorithm exactly as the head stage does.
    if not dry_run or source.exists() or options.algorithm == "inherit":
        options = replace(options, algorithm=resolve_mtp_algorithm(source, options.algorithm))
    target = Path(output_dir).expanduser().resolve() if output_dir else source.with_name(source.name + f"_mtp_w{options.block_bits or options.expert_bits}")
    if not dry_run:
        metadata = inspect_mtp_checkpoint(source, options)
        if metadata:
            return Qwen35MTPRecipeResult(str(source), "skip", "gptqmodel_hf",
                                         {"action": "skipped", "mtp": metadata})
    if source == target or source in target.parents:
        raise ValueError("MTP output must be outside the source checkpoint")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"MTP output already exists: {target}; pass it as model_dir to reuse")
    if type(nsamples) is not int or nsamples <= 0 or type(seqlen) is not int or seqlen < 3:
        raise ValueError("MTP nsamples must be positive and seqlen must be at least 3")
    dataset = Path(calibration_jsonl).expanduser().resolve() if calibration_jsonl else None
    _validate_calibration_source(dataset, calibration_dataset)
    provenance = {"recipe": "gptqmodel.recipes.qwen35_mtp.quantize_qwen35_mtp",
                  "action": "dry_run" if dry_run else "calibrated", "source": str(source),
                  "output_dir": str(target), "mtp": asdict(options), "nsamples": nsamples,
                  "seqlen": seqlen, "calibration_jsonl": str(dataset) if dataset else None,
                  "calibration_dataset": calibration_dataset, "calibration_text_key": calibration_text_key,
                  "device": str(device), "replay_device": str(replay_device), "seed": seed, "dry_run": dry_run,
                  "input_alignment": "main_post_norm_hidden[t], embedding(token[t+1])",
                  "loss": (("linear_output_mse" if options.block_bits else "joint_expert_output_mse")
                           if options.algorithm == "autoround" else "gptq_hessian"),
                  "main_model_and_head": "unchanged"}
    if not dry_run:
        import torch
        import hashlib

        model = Qwen35MTP.from_pretrained(source)
        calibration = _load_calibration(source, dataset, calibration_dataset, calibration_text_key,
                                        nsamples + options.validation_nsamples, seqlen, seed)
        # Cache identity includes actual tokenized samples and local checkpoint
        # file versions. The cache is opt-in and never bypasses source validation.
        identity = {"source": str(source), "files": {
            p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in source.iterdir()
            if p.suffix in (".json", ".safetensors")}, "nsamples": nsamples,
            "validation_nsamples": options.validation_nsamples, "replay_device": str(replay_device),
            "tokens_sha256": hashlib.sha256(b"".join(s["input_ids"].numpy().tobytes() for s in calibration)).hexdigest()}
        cache = Path(calibration_cache).expanduser().resolve() if calibration_cache else None
        if cache is not None and cache.exists():
            saved = torch.load(cache, map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError("MTP calibration cache does not match the source checkpoint and dataset")
            expert_inputs, validation = saved["expert_inputs"], saved["validation"]
            print(f"[mtp] reused calibration inputs: {cache}", flush=True)
        else:
            expert_inputs, validation = collect_mtp_inputs(source, model, calibration, device=device,
                                                          replay_device=replay_device,
                                                          validation_nsamples=options.validation_nsamples)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(cache.suffix + ".tmp")
                torch.save({"identity": identity, "expert_inputs": expert_inputs, "validation": validation}, temporary)
                temporary.replace(cache)
        provenance["calibration_cache"] = str(cache) if cache else None
        payload, metrics = calibrate_mtp_experts(model, expert_inputs, options, device=device)
        provenance["experts"] = metrics
        provenance["validation"] = validate_mtp_payload(model, payload, validation, source, device=device, options=options)
        save_mtp_checkpoint(source, target, payload, options, provenance)
    return Qwen35MTPRecipeResult(str(target), options.algorithm, "gptqmodel_hf", provenance)


def quantize_with_mtp(recipe, options, *, backbone_algorithm):
    """Optional final stage after full backbone/head recipes; no MTP bit search."""
    import tempfile
    import shutil
    from .qwen35 import _normalize_rotation, _repo_root, _MOE_CALIBRATION_JSONL
    from .qwen35_head import autoround_head_calibration_source
    from ..quantization.qwen35_mtp import MTPQuantizationConfig, resolve_mtp_algorithm

    mtp = MTPQuantizationConfig(**options["mtp"])
    mtp = replace(mtp, algorithm=resolve_mtp_algorithm(options["model_dir"], mtp.algorithm,
                                                      backbone_algorithm=backbone_algorithm))
    if _normalize_rotation(options.get("rotation")) or options.get("save_rotated"):
        raise ValueError("MTP calibration requires rotation=false")
    if options.get("topology") not in (None, "moe", "dense"):
        raise ValueError("MTP calibration supports Qwen dense/MoE topology")
    output = Path(options["output_dir"]).expanduser().resolve()
    body = dict(options, mtp=None, output_dir=str(output) + ".main")
    preview = recipe(**dict(body, dry_run=True))
    if (preview.topology == "dense") != (mtp.block_bits is not None):
        raise ValueError("Dense MTP requires block_bits; MoE MTP requires expert_bits")
    source_options = (autoround_head_calibration_source(options["dataset"]) if backbone_algorithm == "autoround"
                      else {"calibration_jsonl": options.get("calibration_jsonl"),
                            "calibration_dataset": options.get("calibration_dataset")})
    if not any(source_options.values()):
        source_options = {"calibration_jsonl": str(_repo_root() / _MOE_CALIBRATION_JSONL)}
    default_nsamples = 512 if backbone_algorithm == "gptq" else 256
    default_seqlen = 1024 if backbone_algorithm == "gptq" else 2048
    mtp_options = dict(model_dir=body["output_dir"], output_dir=str(output), mtp=asdict(mtp),
                       **source_options, nsamples=options.get("nsamples") or default_nsamples,
                       seqlen=options.get("seqlen") or default_seqlen, device=options.get("device") or "cuda:0",
                       replay_device=options.get("mtp_replay_device"),
                       calibration_text_key=options.get("calibration_text_key", "text"), seed=options.get("seed", 42))
    mtp_preview = quantize_qwen35_mtp(**mtp_options, dry_run=True)
    if options.get("dry_run"):
        return replace(preview, output_dir=str(output),
                       provenance={**preview.provenance, "mtp_stage": mtp_preview.provenance})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".main-", dir=output.parent) as tmp:
        body["output_dir"] = str(Path(tmp) / "model")
        result = recipe(**body)
        mtp_options["model_dir"] = result.output_dir
        mtp_result = quantize_qwen35_mtp(**mtp_options)
        log_file = result.provenance.get("log_file")
        if log_file and Path(log_file).is_file():
            saved_log = output / "main_quantization.log"
            shutil.copy2(log_file, saved_log)
            result = replace(result, provenance={**result.provenance, "log_file": str(saved_log)})
        search_report = Path(result.output_dir + ".auto_mixed_precision_search.json")
        if search_report.is_file():
            shutil.copy2(search_report, Path(str(output) + ".auto_mixed_precision_search.json"))
    result = replace(result, output_dir=mtp_result.output_dir,
                     provenance={**result.provenance, "mtp_stage": mtp_result.provenance})
    if backbone_algorithm == "autoround":
        from .qwen35_autoround import _write_recipe_provenance
    else:
        from .qwen35 import _write_recipe_provenance
    _write_recipe_provenance(result.output_dir, result.provenance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--mtp", type=json.loads, default={})
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--calibration-jsonl")
    data.add_argument("--calibration-dataset")
    parser.add_argument("--calibration-text-key", default="text")
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-cache")
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(asdict(quantize_qwen35_mtp(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
