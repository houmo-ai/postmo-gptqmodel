"""Stable GPTQ recipe for text-only Qwen3-Next checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..utils.missing_tensors import count_checkpoint_tensors_with_prefix


@dataclass(frozen=True)
class Qwen3NextQuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    has_mtp: bool
    max_quant_layers: int | None
    provenance: dict[str, Any]


def quantize_qwen3_next(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    artifact_format: str = "gptqmodel_hf",
    max_quant_layers: int | None = None,
    rotation: str | bool | None = False,
    bits: int = 4,
    self_attn_bits: int = 8,
    linear_attn_bits: int = 8,
    dense_mlp_bits: int = 4,
    expert_bits: int = 4,
    shared_expert_bits: int = 8,
    group_size: int = 64,
    batch_size: int = 1,
    nsamples: int = 256,
    seqlen: int = 1024,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    device: str | None = None,
    device_map: str | None = "auto",
    infer_device_map: str | None = None,
    auto_forward_data_parallel: bool = False,
    trust_remote_code: bool = True,
    check_saved_precision: bool = False,
    min_logit_cosine: float = 0.99,
    require_top1_match: bool = True,
    skip_ppl: bool = False,
    max_new_tokens: int = 128,
    prompt: str | None = None,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Qwen3NextQuantRecipeResult:
    """Quantize Qwen3-Next through the maintained example entrypoint."""

    method = str(method).lower()
    if method != "gptq":
        raise ValueError("method must be 'gptq'")
    if artifact_format != "gptqmodel_hf":
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if max_quant_layers is not None and max_quant_layers < 0:
        raise ValueError("max_quant_layers must be >= 0")
    if rotation is False:
        rotation = None
    if rotation is not None:
        raise ValueError("Qwen3-Next MTP preservation currently requires rotation=False")

    has_mtp = count_checkpoint_tensors_with_prefix(str(model_dir), "mtp.") > 0

    command = [
        sys.executable,
        str(_repo_root() / "examples/quantization/examples/example_qwen3_next.py"),
        "--model",
        str(model_dir),
        "--out",
        str(output_dir),
        "--bits",
        str(bits),
        "--self-attn-bits",
        str(self_attn_bits),
        "--linear-attn-bits",
        str(linear_attn_bits),
        "--dense-mlp-bits",
        str(dense_mlp_bits),
        "--expert-bits",
        str(expert_bits),
        "--shared-expert-bits",
        str(shared_expert_bits),
        "--group-size",
        str(group_size),
        "--batch-size",
        str(batch_size),
        "--nsamples",
        str(nsamples),
        "--seqlen",
        str(seqlen),
        "--calibration-text-key",
        calibration_text_key,
        "--min-logit-cosine",
        str(min_logit_cosine),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if max_quant_layers is not None:
        command.extend(["--max-quant-layers", str(max_quant_layers)])
    if rotation:
        command.extend(["--rotation", str(rotation)])
    if calibration_jsonl:
        command.extend(["--calibration-jsonl", calibration_jsonl])
    if device_map:
        command.extend(["--device-map", device_map])
    elif device:
        command.extend(["--device", device])
    if infer_device_map:
        command.extend(["--infer-device-map", infer_device_map])
    command.append(
        "--auto-forward-data-parallel"
        if auto_forward_data_parallel
        else "--no-auto-forward-data-parallel"
    )
    if trust_remote_code:
        command.append("--trust-remote-code")
    if check_saved_precision:
        command.append("--check-saved-precision")
    command.append("--require-top1-match" if require_top1_match else "--no-require-top1-match")
    if skip_ppl:
        command.append("--skip-ppl")
    if prompt:
        command.extend(["--prompt", prompt])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    provenance = {
        "recipe": "gptqmodel.recipes.qwen3_next.quantize_qwen3_next",
        "backend": "examples/quantization/examples/example_qwen3_next.py",
        "command": command,
        "dry_run": dry_run,
    }
    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update({str(key): str(value) for key, value in env.items()})
        subprocess.run(command, cwd=_repo_root(), env=run_env, check=True)

    return Qwen3NextQuantRecipeResult(
        output_dir=str(Path(output_dir).expanduser().resolve()),
        method=method,
        artifact_format=artifact_format,
        model_family="qwen3_next",
        has_mtp=has_mtp,
        max_quant_layers=max_quant_layers,
        provenance=provenance,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
