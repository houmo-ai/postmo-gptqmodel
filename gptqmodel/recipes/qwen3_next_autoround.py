"""Qwen3-Next's vendored AutoRound backend (no ModelZoo dependency).

The public entrypoint remains ``quantize_qwen3_next(method="autoround")``.
Loading patches run in a subprocess so Transformers' global MoE replacements
cannot leak into a caller's later GPTQ/export work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DTYPES = ("int_sym_xh", "int_sym", "int_sym_gptq", "int")
LOG_NAME = "qwen3_next_autoround_quantize.log"
PROVENANCE_NAME = "qwen3_next_autoround_recipe_provenance.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, choices=(64,), default=64)
    for name, default in (
        ("self-attn", 8),
        ("linear-attn", 8),
        ("dense-mlp", 4),
        ("expert", 4),
        ("shared-expert", 8),
    ):
        parser.add_argument(
            f"--{name}-bits", type=int, choices=(4, 8, 16), default=default
        )
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-type", choices=DTYPES, default="int_sym_xh")
    for name, default in (
        ("iters", 400),
        ("nsamples", 256),
        ("seqlen", 2048),
        ("batch-size", 8),
        ("gradient-accumulate-steps", 1),
        ("seed", 42),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--calibration-jsonl")
    parser.add_argument("--device-map", default="0")
    parser.add_argument(
        "--low-gpu-mem-usage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--format", choices=("auto_round:gptqmodel",), default="auto_round:gptqmodel"
    )
    return parser


def validate_args(args) -> None:
    if args.data_type.startswith("int_sym") and not args.sym:
        raise ValueError(f"data_type={args.data_type!r} requires sym=True")
    for name in ("nsamples", "seqlen", "batch_size", "gradient_accumulate_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.iters < 0:
        raise ValueError("iters must be >= 0")
    if Path(args.model).expanduser().resolve() == Path(args.out).expanduser().resolve():
        raise ValueError("output_dir must differ from model_dir")


def run_recipe(
    *, model_dir, output_dir, dry_run=False, extra_args=None, env=None, **options
) -> dict:
    root = Path(__file__).resolve().parents[2]
    # Resolve caller-relative paths before the child changes cwd to GPTQModel.
    # ModelZoo may be checked out somewhere completely different.
    model_dir = str(Path(model_dir).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())
    if options.get("calibration_jsonl"):
        options["calibration_jsonl"] = str(
            Path(options["calibration_jsonl"]).expanduser().resolve()
        )
    dataset = options.get("dataset")
    if dataset and (
        Path(dataset).expanduser().exists()
        or str(dataset).endswith((".json", ".jsonl"))
    ):
        options["dataset"] = str(Path(dataset).expanduser().resolve())
    argv = ["--model", model_dir, "--out", output_dir]
    for key, value in options.items():
        if value is None:
            continue
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(f"--{'' if value else 'no-'}{flag}")
        else:
            argv.extend([f"--{flag}", str(value)])
    if extra_args:
        # Keep recipe paths/provenance authoritative. AutoRound exposes named
        # options above; the old GPTQ path retains its existing escape hatch.
        raise ValueError(
            "AutoRound extra_args are unsupported; use named recipe parameters"
        )
    args = build_parser().parse_args(argv)
    validate_args(args)
    backend = "examples/quantization/examples/example_qwen3_next_autoround.py"
    command = [sys.executable, str(root / backend), *argv]
    provenance = {
        "recipe": "gptqmodel.recipes.qwen3_next.quantize_qwen3_next",
        "backend": backend,
        "command": command,
        "data_type": args.data_type,
        "parameters": vars(args),
        "dry_run": bool(dry_run),
        "log_file": str(Path(output_dir) / LOG_NAME),
    }
    if dry_run:
        return provenance
    _prepare_output(args)
    run_env = os.environ.copy()
    if env:
        run_env.update({str(key): str(value) for key, value in env.items()})
    with open(provenance["log_file"], "w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=root,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    # The backend adds actual load/quantization/MTP details before it exits.
    record = Path(output_dir) / PROVENANCE_NAME
    provenance.update(json.loads(record.read_text(encoding="utf-8")))
    record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def _prepare_output(args) -> None:
    target = Path(args.out)
    if target.exists() and any(path.name != LOG_NAME for path in target.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty quantization output: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)


def build_layer_config(model, args) -> dict:
    """Explicit allowlist, matching Qwen3NextGPTQ.module_tree's FP exclusions."""
    import torch

    policies = (
        (r"self_attn\.(q|k|v|o)_proj", args.self_attn_bits),
        (r"linear_attn\.(in_proj_qkvz|out_proj)", args.linear_attn_bits),
        (r"mlp\.(gate|up|down)_proj", args.dense_mlp_bits),
        (r"mlp\.experts\.\d+\.(gate|up|down)_proj", args.expert_bits),
        (r"mlp\.shared_expert\.(gate|up|down)_proj", args.shared_expert_bits),
    )
    config = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        bits = 16
        for pattern, candidate in policies:
            if re.fullmatch(r"model\.layers\.\d+\." + pattern, name):
                bits = candidate
                break
        config[name] = (
            {
                "bits": bits,
                "group_size": args.group_size,
                "sym": args.sym,
                "data_type": args.data_type,
            }
            if bits < 16
            else {"bits": 16, "data_type": "fp"}
        )
    # Never proceed with fused 3-D experts: their weights would miss this
    # Linear-only allowlist and silently remain floating point.
    for layer in model.model.layers:
        if hasattr(layer.mlp, "experts"):
            experts = layer.mlp.experts
            expert_modules = {
                name: child
                for name, child in experts.named_children()
                if name.isdigit()
            }
            if set(expert_modules) != {
                str(index) for index in range(model.config.num_experts)
            } or not all(
                isinstance(getattr(expert, projection, None), torch.nn.Linear)
                for expert in expert_modules.values()
                for projection in ("gate_proj", "up_proj", "down_proj")
            ):
                raise ValueError(
                    "Qwen3-Next AutoRound requires unfused gate/up/down Linear experts"
                )
    if not any(value["bits"] < 16 for value in config.values()):
        raise ValueError("No Qwen3-Next quantizable projections found")
    return config


def quantize(args) -> dict:
    validate_args(args)
    _prepare_output(args)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import transformers
    from auto_round import AutoRound
    from auto_round.modeling.unfused_moe import apply_model_monkey_patches
    from auto_round.version import __version__ as autoround_version
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

    from gptqmodel.utils.missing_tensors import (
        copy_missing_tensors_from_source,
        count_checkpoint_tensors_with_prefix,
        validate_tensor_prefix_parity,
    )

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if config.model_type != "qwen3_next":
        raise ValueError(f"Expected model_type='qwen3_next', got {config.model_type!r}")
    if getattr(config, "quantization_config", None):
        raise ValueError(
            "Qwen3-Next AutoRound expects an unquantized source checkpoint"
        )
    if not apply_model_monkey_patches(
        args.model, trust_remote_code=args.trust_remote_code
    ):
        raise ValueError(
            "Cannot load Qwen3-Next with unfused experts; check checkpoint layout/Transformers version"
        )
    set_seed(args.seed)
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": "auto",
        "device_map": None,
    }
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        load_kwargs["attn_implementation"] = "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.model, output_loading_info=True, **load_kwargs
    )
    missing = loading_info.get("missing_keys", [])
    unexpected = [
        name
        for name in loading_info.get("unexpected_keys", [])
        if not name.startswith("mtp.")
    ]
    if (
        missing
        or unexpected
        or loading_info.get("mismatched_keys")
        or loading_info.get("error_msgs")
    ):
        raise RuntimeError(
            f"Incomplete Qwen3-Next checkpoint load: missing={list(missing)[:8]}, "
            f"unexpected={unexpected[:8]}; refusing to quantize partially initialized weights"
        )
    model.eval()
    layer_config = build_layer_config(model, args)
    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        data_type=args.data_type,
        layer_config=layer_config,
        iters=args.iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        dataset=args.calibration_jsonl or args.dataset,
        seed=args.seed,
        device_map=args.device_map,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        enable_deterministic_algorithms=args.deterministic,
    )
    # The model was loaded with AutoRound's own patch above; retain its supported
    # low-CPU-memory offload path, as when AutoRound loads a model-name string.
    ar.is_model_patched = True
    ar.quantize()
    ar.save_quantized(args.out, format=args.format)
    copied = copy_missing_tensors_from_source(
        args.model, args.out, tensor_name_prefixes=("mtp.",)
    )
    mtp_count = count_checkpoint_tensors_with_prefix(args.model, "mtp.")
    if mtp_count:
        validate_tensor_prefix_parity(args.model, args.out, prefix="mtp.")
    details = {
        "transformers_version": transformers.__version__,
        "autoround_version": autoround_version,
        "attention_implementation": model.config._attn_implementation,
        "mtp_tensors_copied": copied,
        "mtp_tensor_count": mtp_count,
        "layer_config": layer_config,
    }
    (Path(args.out) / PROVENANCE_NAME).write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    return details


def main(argv=None) -> None:
    quantize(build_parser().parse_args(argv))
