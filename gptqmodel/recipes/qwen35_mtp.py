# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Complete Qwen3.5/3.6 MoE MTP experts on HF or already quantized checkpoints."""
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


def collect_mtp_inputs(source, model, calibration, *, device, validation_nsamples):
    """Replay the supplied main model, then align hidden[t] with token[t+1]."""
    import torch
    from .. import GPTQModel
    from ..utils.backend import BACKEND

    from ..quantization.lm_head import read_checkpoint_config
    config = read_checkpoint_config(source)
    if config.get("quantization_config"):
        main = GPTQModel.load(str(source), device=device, backend=BACKEND.TORCH, torch_dtype=torch.float16)
        owner = main.model
    else:
        from transformers import AutoModelForImageTextToText
        owner = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float16, device_map={"": device})
        main = owner
    language = owner.model.language_model
    language.eval()
    # No logits are needed: language.forward already applies its final norm.
    model.to(device).eval()
    expert_inputs = []
    validation = []
    captured = []

    def capture(module, args):
        captured.append(args[0].detach().reshape(-1, model.config.hidden_size).cpu())

    handle = model.layers[0].mlp.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for index, sample in enumerate(calibration):
                ids = sample["input_ids"].to(device)
                mask = sample.get("attention_mask", torch.ones_like(ids)).to(device)
                if ids.shape[0] != 1:
                    raise ValueError("MTP calibration expects individual unpadded sequences")
                ids = ids[:, mask[0].bool()]
                if ids.shape[1] < 3:
                    raise ValueError("MTP calibration needs at least three non-padding tokens")
                hidden = language(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                embeddings = owner.get_input_embeddings()(ids[:, 1:])
                embeddings = embeddings.to(model.fc.weight.dtype)
                mtp_hidden = hidden[:, :-1].to(model.fc.weight.dtype)
                reference = model(embeddings, mtp_hidden)
                if index < len(calibration) - validation_nsamples:
                    expert_inputs.extend(captured)
                else:
                    validation.append({"embedding": embeddings.cpu(), "hidden": mtp_hidden.cpu(),
                                       "reference": reference.cpu(), "main_next_hidden": hidden[:, 1:].cpu()})
                captured.clear()
                print(f"[mtp] collected sequence {index + 1}/{len(calibration)}, tokens={ids.shape[1] - 1}", flush=True)
    finally:
        handle.remove()
        model.cpu()
        del main, owner, language
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return expert_inputs, validation


def quantize_qwen35_mtp(*, model_dir, mtp, output_dir=None, calibration_jsonl=None,
                       calibration_dataset=None, calibration_text_key="text", nsamples=256,
                       seqlen=2048, device="cuda:0", seed=42, dry_run=False, calibration_cache=None):
    from ..quantization.qwen35_mtp import (
        MTPQuantizationConfig, Qwen35MTP, calibrate_mtp_experts, inspect_mtp_checkpoint, save_mtp_checkpoint,
        validate_mtp_payload, resolve_mtp_algorithm,
    )
    from .qwen35_head import _load_calibration, _validate_calibration_source

    options = MTPQuantizationConfig(**{"seed": seed, **mtp})
    source = Path(model_dir).expanduser().resolve()
    # Full-recipe previews refer to a not-yet-created backbone checkpoint;
    # that caller already resolves the algorithm exactly as the head stage does.
    if not dry_run or source.exists() or options.algorithm == "inherit":
        options = replace(options, algorithm=resolve_mtp_algorithm(source, options.algorithm))
    target = Path(output_dir).expanduser().resolve() if output_dir else source.with_name(source.name + "_mtp_w4")
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
                  "device": str(device), "seed": seed, "dry_run": dry_run,
                  "input_alignment": "main_post_norm_hidden[t], embedding(token[t+1])",
                  "loss": "joint_expert_output_mse" if options.algorithm == "autoround" else "gptq_hessian",
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
            "validation_nsamples": options.validation_nsamples,
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
                                                          validation_nsamples=options.validation_nsamples)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(cache.suffix + ".tmp")
                torch.save({"identity": identity, "expert_inputs": expert_inputs, "validation": validation}, temporary)
                temporary.replace(cache)
        provenance["calibration_cache"] = str(cache) if cache else None
        payload, metrics = calibrate_mtp_experts(model, expert_inputs, options, device=device)
        provenance["experts"] = metrics
        provenance["validation"] = validate_mtp_payload(model, payload, validation, source, device=device)
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
    if options.get("topology") not in (None, "moe"):
        raise ValueError("MTP calibration supports MoE topology only")
    output = Path(options["output_dir"]).expanduser().resolve()
    body = dict(options, mtp=None, output_dir=str(output) + ".main")
    preview = recipe(**dict(body, dry_run=True))
    if preview.topology != "moe":
        raise ValueError("MTP calibration supports MoE topology only")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-cache")
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(asdict(quantize_qwen35_mtp(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
