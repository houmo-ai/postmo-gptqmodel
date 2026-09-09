# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 head calibration for GPTQ and AutoRound checkpoints."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Qwen35HeadRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    provenance: dict


def _validate_calibration_source(dataset, calibration_dataset):
    if bool(dataset) == bool(calibration_dataset):
        raise ValueError("Head calibration requires exactly one of calibration_jsonl or calibration_dataset")
    if dataset is not None and not dataset.is_file():
        raise FileNotFoundError(f"Head calibration JSONL does not exist: {dataset}")


def _load_calibration(source, dataset, calibration_dataset, text_key, nsamples, seqlen, seed):
    from transformers import AutoTokenizer

    _validate_calibration_source(dataset, calibration_dataset)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if dataset is not None:
        with dataset.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        random.Random(seed).shuffle(records)
    else:
        from datasets import load_dataset
        records = load_dataset(calibration_dataset, split="train").shuffle(seed=seed)
    calibration = []
    for record in records:
        text = record if isinstance(record, str) else record.get(text_key)
        if not isinstance(text, str):
            raise ValueError(f"Calibration field {text_key!r} must contain text")
        encoded = tokenizer(text, truncation=True, max_length=seqlen, return_tensors="pt")
        if encoded["input_ids"].shape[1] < min(seqlen, 16):
            continue
        calibration.append({key: encoded[key] for key in ("input_ids", "attention_mask")})
        if len(calibration) == nsamples:
            break
    if len(calibration) != nsamples:
        raise ValueError(f"Only {len(calibration)} usable calibration samples; requested {nsamples}")
    return calibration


def quantize_qwen35_head(*, model_dir, head, output_dir=None, calibration_jsonl=None, calibration_dataset=None,
                        calibration_text_key="text", nsamples=256, seqlen=2048,
                        batch_size=1, device="cuda:0", seed=42, dry_run=False):
    from ..quantization.lm_head import HeadQuantizationConfig, inspect_checkpoint_head, resolve_head_algorithm

    if head is None:
        raise ValueError("Head calibration requires head configuration")
    head = HeadQuantizationConfig(**{"seed": seed, **head})
    source = Path(model_dir).expanduser().resolve()
    if head.bits >= 8:
        return Qwen35HeadRecipeResult(str(source), "skip", "gptqmodel_hf",
                                     {"action": "skipped", "reason": "W8 uses the existing export quantization"})
    if not dry_run:
        skipped = inspect_checkpoint_head(source, head)
        if skipped is not None:
            return Qwen35HeadRecipeResult(str(source), "skip", "gptqmodel_hf",
                                         {"action": "skipped", "operation": skipped})
    algorithm = resolve_head_algorithm(model_dir, head.algorithm)
    head = replace(head, algorithm=algorithm)
    if min(nsamples, seqlen, batch_size) <= 0:
        raise ValueError("nsamples, seqlen and batch_size must be positive")
    target = Path(output_dir).expanduser().resolve() if output_dir is not None else source.with_name(
        source.name + f"_head_w{head.bits}")
    if source == target or source in target.parents:
        raise ValueError("Head output must be outside the source checkpoint")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Head output already exists: {target}. Pass this checkpoint as model_dir to reuse it")
    dataset = Path(calibration_jsonl).expanduser().resolve() if calibration_jsonl else None
    if dry_run and (dataset is not None or calibration_dataset):
        _validate_calibration_source(dataset, calibration_dataset)
    command = [sys.executable, "-m", "gptqmodel.recipes.qwen35_head",
               "--model-dir", str(source), "--output-dir", str(target),
               "--head", json.dumps(asdict(head)),
               "--calibration-text-key", calibration_text_key, "--nsamples", str(nsamples),
               "--seqlen", str(seqlen), "--batch-size", str(batch_size),
               "--device", str(device), "--seed", str(seed)]
    if dataset is not None:
        command.extend(["--calibration-jsonl", str(dataset)])
    elif calibration_dataset:
        command.extend(["--calibration-dataset", str(calibration_dataset)])
    provenance = {"recipe": "gptqmodel.recipes.qwen35_head.quantize_qwen35_head",
                  "command": command, "head": asdict(head), "dry_run": dry_run,
                  "nsamples": nsamples, "seqlen": seqlen, "seed": seed,
                  "calibration_jsonl": str(dataset) if dataset is not None else None,
                  "calibration_dataset": calibration_dataset,
                  "action": "check_head"}
    method = algorithm
    if not dry_run:
        import torch
        from .. import GPTQModel
        from ..utils.backend import BACKEND

        _validate_calibration_source(dataset, calibration_dataset)
        if output_dir is None:
            warnings.warn(
                f"Requested head W{head.bits} (<8), but {source} has a floating head. "
                f"GPTQModel will calibrate only the head using {algorithm} and save a new checkpoint to {target}. "
                f"Use {target} directly with the matching W{head.bits} head YAML on subsequent exports.",
                UserWarning, stacklevel=2)
        # Keep the quantized backbone on the host; the common looper moves one
        # decoder block at a time to the requested calibration device.
        model = GPTQModel.load(str(source), device="cpu", backend=BACKEND.TORCH)
        model.quantize_config.device = torch.device(device)
        calibration = _load_calibration(
            source, dataset, calibration_dataset, calibration_text_key, nsamples, seqlen, seed)
        model.requantize(calibration, head=asdict(head), batch_size=batch_size)
        provenance["action"] = model._lm_head_last_operation["action"]
        provenance["operation"] = model._lm_head_last_operation
        model.save(str(target))
        (target / "qwen35_head_recipe_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return Qwen35HeadRecipeResult(str(target), method, "gptqmodel_hf", provenance)


def quantize_with_head(recipe, options, *, backbone_algorithm):
    """Run a full backbone recipe, then reuse the same endpoint calibration API."""
    from ..quantization.lm_head import HeadQuantizationConfig, read_checkpoint_config

    from .qwen35 import _normalize_rotation, _repo_root

    if _normalize_rotation(options.get("rotation")) or options.get("save_rotated"):
        raise ValueError("Full quantization with head calibration currently requires rotation=false")
    if read_checkpoint_config(options["model_dir"]).get("quantization_config"):
        raise ValueError("Input is already quantized; use the existing skip-quantization option and head completion")
    head = HeadQuantizationConfig(**{"seed": options.get("seed", 42),
        "iters": options.get("iters") or 400, "data_type": options.get("data_type"), **options["head"]})
    if head.algorithm == "inherit":
        head = replace(head, algorithm=backbone_algorithm)
    original_output = str(Path(options["output_dir"]).expanduser().resolve())
    body_options = dict(options, head=None, output_dir=str(original_output) + ".backbone")
    body_output = Path(body_options["output_dir"]).expanduser().resolve()
    preview = recipe(**dict(body_options, dry_run=True))
    calibration_jsonl = options.get("calibration_jsonl")
    calibration_dataset = options.get("calibration_dataset")
    if backbone_algorithm == "autoround":
        source_options = autoround_head_calibration_source(options.get("dataset"))
        calibration_jsonl = source_options.get("calibration_jsonl")
        calibration_dataset = source_options.get("calibration_dataset")
    if not calibration_jsonl and not calibration_dataset:
        from .qwen35 import _DENSE_CALIBRATION_JSONL, _MOE_CALIBRATION_JSONL
        calibration_jsonl = str(_repo_root() / (
            _MOE_CALIBRATION_JSONL if preview.topology == "moe" else _DENSE_CALIBRATION_JSONL))
    dataset_path = Path(calibration_jsonl).expanduser() if calibration_jsonl else None
    if dataset_path is not None and not dataset_path.is_absolute() and not dataset_path.is_file():
        dataset_path = _repo_root() / dataset_path
    default_nsamples = 512 if backbone_algorithm == "gptq" and preview.topology == "moe" else 256
    default_seqlen = 1024 if backbone_algorithm == "gptq" else 2048
    head_options = dict(model_dir=str(body_output), output_dir=original_output, head=asdict(head),
                        calibration_jsonl=str(dataset_path.resolve()) if dataset_path is not None else None,
                        calibration_dataset=calibration_dataset,
                        calibration_text_key=options.get("calibration_text_key", "text"),
                        nsamples=options.get("nsamples") or default_nsamples,
                        seqlen=options.get("seqlen") or default_seqlen,
                        batch_size=1, device=options.get("device") or "cuda:0", seed=options.get("seed", 42))
    # Validate the complete second stage before starting an expensive backbone job.
    quantize_qwen35_head(**head_options, dry_run=True)
    if options.get("dry_run", False):
        result = preview
        head_result = quantize_qwen35_head(**head_options, dry_run=True)
    else:
        # The two internal stages publish one final checkpoint at output_dir.
        # There is no additional workflow mode or persistent backbone artifact.
        Path(original_output).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=Path(original_output).name + ".backbone-",
                                         dir=Path(original_output).parent) as temporary:
            body_options["output_dir"] = str(Path(temporary) / "model")
            result = recipe(**body_options)
            head_options["model_dir"] = result.output_dir
            head_result = quantize_qwen35_head(**head_options)
            log_file = result.provenance.get("log_file")
            if log_file and Path(log_file).is_file():
                saved_log = Path(head_result.output_dir) / "backbone_quantization.log"
                shutil.copy2(log_file, saved_log)
                result = replace(result, provenance={**result.provenance, "log_file": str(saved_log)})
    result = replace(result, output_dir=head_result.output_dir,
                     provenance={**result.provenance, "head_stage": head_result.provenance})
    if not options.get("dry_run", False):
        if backbone_algorithm == "autoround":
            from .qwen35_autoround import _write_recipe_provenance
        else:
            from .qwen35 import _write_recipe_provenance
        _write_recipe_provenance(result.output_dir, result.provenance)
    return result


def autoround_head_calibration_source(dataset):
    """Keep AutoRound's local JSONL and HF dataset-name API semantics."""
    path = Path(dataset).expanduser()
    if path.is_file() or path.is_absolute() or path.suffix.lower() in (".json", ".jsonl"):
        return {"calibration_jsonl": str(path)}
    return {"calibration_dataset": dataset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--head", type=json.loads, required=True)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration-jsonl")
    calibration.add_argument("--calibration-dataset")
    parser.add_argument("--calibration-text-key", default="text")
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    result = quantize_qwen35_head(**vars(parser.parse_args()))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
