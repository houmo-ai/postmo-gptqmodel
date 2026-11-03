"""Calibrate a MiniCPM head while retaining its float-QDQ text backbone."""
from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from dataclasses import asdict, replace
from pathlib import Path


def _stage_qdq_checkpoint(source, tokenizer_dir, destination, algorithm):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from ..quantization import QuantizeConfig
    from ..quantization.config import FORMAT

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("MiniCPM head input must be a Qwen3.5 Dense/MoE text backbone")
    state = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if "lm_head.weight" not in state or any(not t.is_floating_point() for t in state.values()):
        raise ValueError("MiniCPM QDQ input requires floating backbone tensors and lm_head.weight")
    head = state.pop("lm_head.weight")
    if tuple(head.shape) != (config["vocab_size"], config["hidden_size"]):
        raise ValueError("MiniCPM head shape does not match its text configuration")
    # GPTQModel materializes per-expert Linears when loading a text model.
    # Split contiguous views exactly; no rounding or casting of QDQ weights.
    for name in tuple(state):
        if not name.endswith(".mlp.experts.gate_up_proj"):
            continue
        prefix = name.removesuffix("gate_up_proj")
        fused = state.pop(name)
        down = state.pop(prefix + "down_proj")
        for expert in range(fused.shape[0]):
            gate, up = fused[expert].chunk(2, dim=0)
            for suffix, tensor in (("gate_proj", gate), ("up_proj", up), ("down_proj", down[expert])):
                state[prefix + f"{expert}.{suffix}.weight"] = tensor.contiguous()
    state = {name: tensor.contiguous() for name, tensor in state.items()}
    save_file(state, destination / "model-backbone.safetensors", metadata={"format": "pt"})
    save_file({"lm_head.weight": head.contiguous()}, destination / "model-head-float.safetensors",
              metadata={"format": "pt"})
    mapping = dict.fromkeys(state, "model-backbone.safetensors")
    mapping["lm_head.weight"] = "model-head-float.safetensors"
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())
                     + head.numel() * head.element_size()}, "weight_map": mapping,
    }, indent=2) + "\n")
    qcfg = QuantizeConfig(bits=8, group_size=64, sym=True, desc_act=False,
                         format=FORMAT.GPTQ_V2, lm_head=False,
                         dynamic={"-:^(?!lm_head$).*": {}}).to_dict()
    provenance_file = source / "minicpm_v_4_7_quant_recipe_provenance.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.is_file() else {}
    qcfg["data_type"] = provenance.get("precision", {}).get(
        "data_type", "int_sym_xh" if algorithm == "autoround" else "int")
    qcfg.setdefault("meta", {})["float_qdq_backbone"] = {
        "model_family": "minicpm_v_4_7", "artifact_format": "float_causal_bin",
        "source": str(source), "producer": provenance,
    }
    if provenance.get("method", algorithm) == "autoround":
        qcfg["meta"]["auto_round"] = {"auto_round_enabled": True}
    config.update(quantization_config=qcfg, tie_word_embeddings=False)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True).save_pretrained(destination)


def quantize_minicpm_v47_head(*, model_dir, tokenizer_dir, head, output_dir=None,
                             calibration_jsonl=None, calibration_dataset=None,
                             nsamples=256, seqlen=2048, batch_size=1, device="cuda:0",
                             seed=42, dry_run=False):
    from ..quantization.lm_head import HeadQuantizationConfig
    from .qwen35_head import Qwen35HeadRecipeResult, quantize_qwen35_head

    requested = HeadQuantizationConfig(**{"seed": seed, **head})
    source = Path(model_dir).expanduser().resolve()
    if requested.bits >= 8:
        return Qwen35HeadRecipeResult(str(source), "skip", "float_causal_bin", {"action": "skipped"})
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"MiniCPM head checkpoint config does not exist: {source / 'config.json'}")
    common = dict(head=asdict(requested), output_dir=output_dir, calibration_jsonl=calibration_jsonl,
                  calibration_dataset=calibration_dataset, nsamples=nsamples, seqlen=seqlen,
                  batch_size=batch_size, device=device, seed=seed, dry_run=dry_run)
    if (source / "model.safetensors.index.json").is_file():
        config = json.loads((source / "config.json").read_text())
        marker = ((config.get("quantization_config") or {}).get("meta") or {}).get("float_qdq_backbone", {})
        if marker.get("model_family") != "minicpm_v_4_7":
            raise ValueError("Expected a MiniCPM float-QDQ checkpoint")
        return quantize_qwen35_head(model_dir=str(source), **common)
    if not (source / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"MiniCPM QDQ backbone weights do not exist: {source / 'pytorch_model.bin'}")
    algorithm = requested.algorithm
    if algorithm == "inherit":
        provenance = source / "minicpm_v_4_7_quant_recipe_provenance.json"
        algorithm = json.loads(provenance.read_text()).get("method") if provenance.is_file() else None
        if algorithm not in {"gptq", "autoround"}:
            raise ValueError("Cannot infer the QDQ producer; set head.algorithm explicitly")
    requested = replace(requested, algorithm=algorithm)
    target = Path(output_dir).expanduser().resolve() if output_dir else source.with_name(
        source.name + f"_head_w{requested.bits}")
    if target == source or source in target.parents:
        raise ValueError("Head output must be outside the source checkpoint")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Head output already exists: {target}; reuse it as model_dir")
    if dry_run:
        return Qwen35HeadRecipeResult(str(target), algorithm, "gptqmodel_hf", {
            "action": "planned", "head": asdict(requested), "source_checkpoint": str(source)})
    if output_dir is None:
        warnings.warn(f"Calibrating only MiniCPM head W{requested.bits}; saving new checkpoint to {target}",
                      UserWarning, stacklevel=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=target.name + ".qdq-", dir=target.parent) as temporary:
        stage = Path(temporary)
        _stage_qdq_checkpoint(source, tokenizer_dir, stage, algorithm)
        common.update(head=asdict(requested), output_dir=str(target))
        result = quantize_qwen35_head(model_dir=str(stage), **common)
    return result


def quantize_with_minicpm_head(recipe, options, *, algorithm):
    """Publish one final checkpoint after backbone and head calibration."""
    from ..quantization.lm_head import HeadQuantizationConfig
    from .qwen35_head import autoround_head_calibration_source
    from .minicpm_v_4_7 import _write_provenance

    requested = HeadQuantizationConfig(**{
        "seed": options.get("seed", options.get("sampling_seed")) or 0,
        "iters": options.get("iters", 400), **options["head"],
    })
    body_options = dict(options, head=None)
    if requested.bits >= 8:
        return recipe(**body_options)
    if requested.algorithm == "inherit":
        requested = replace(requested, algorithm=algorithm)
    target = Path(options["output_dir"]).expanduser().resolve()
    source = Path(options["model_dir"]).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("MiniCPM quantization output must be outside the source checkpoint")
    if options.get("calibration_data") is not None:
        body_options["calibration_data"] = list(options["calibration_data"])
    preview = recipe(**dict(body_options, dry_run=True))
    planned = replace(preview, output_dir=str(target), artifact_format="gptqmodel_hf",
                      precision={**preview.precision, "head_bits": requested.bits},
                      provenance={**preview.provenance, "head": asdict(requested),
                                  "stages": ["backbone", "head"], "model_dir": options["model_dir"]})
    if options.get("dry_run", False):
        return planned
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"MiniCPM head output directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=target.name + ".backbone-", dir=target.parent) as temporary:
        body_options["output_dir"] = str(Path(temporary) / "body")
        calibration = {}
        if algorithm == "autoround":
            calibration = autoround_head_calibration_source(options["dataset"])
        elif body_options.get("calibration_data") is not None:
            path = Path(temporary) / "calibration.jsonl"
            path.write_text("\n".join(json.dumps({"text": text})
                                      for text in body_options["calibration_data"]) + "\n")
            calibration = {"calibration_jsonl": str(path)}
        else:
            calibration = {"calibration_jsonl": options.get("calibration_jsonl")}
        # Check calibration availability before running an expensive backbone job.
        from .qwen35_head import _validate_calibration_source
        dataset = calibration.get("calibration_jsonl")
        _validate_calibration_source(Path(dataset) if dataset else None, calibration.get("calibration_dataset"))
        body = recipe(**body_options)
        result = quantize_minicpm_v47_head(
            model_dir=body.output_dir, tokenizer_dir=options["model_dir"], output_dir=str(target),
            head=asdict(requested), nsamples=options["nsamples"], seqlen=options["seqlen"],
            device=options["device"], seed=requested.seed, **calibration,
        )
    completed = replace(planned, provenance={**planned.provenance, "dry_run": False,
                                            "head_stage": result.provenance})
    _write_provenance(target, completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head", type=json.loads, required=True)
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = quantize_minicpm_v47_head(**vars(args))
    print(json.dumps({"output_dir": result.output_dir, "method": result.method}))


if __name__ == "__main__":
    main()
