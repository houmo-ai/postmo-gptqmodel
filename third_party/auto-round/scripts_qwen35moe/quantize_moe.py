"""Qwen3.5-MoE (35B-A3B) quantization script with 4 operational modes.

Adapted from scripts/quantize.py for MoE architecture.

Modes:
  1. llm-only       : Only quantize LLM (W4G64, no rotation)
  2. rotate-merger   : Rotate LLM + vision merger output, quantize LLM (W4G64)
  3. rotate-full     : Rotate LLM + full vision encoder, quantize LLM (W4G64)
  4. rotate-quant-all: Rotate LLM + full vision, quantize vision W8G64 + LLM W4G64

Key MoE differences from dense Qwen3.5:
  - Architecture: Qwen3_5MoeForConditionalGeneration (256 experts, 8 active)
  - RMSNorm uses (1 + weight) scaling
  - MLP has gate (router), shared_expert, per-expert gate/up/down_proj
  - hidden_size=2048 (power of 2 → exact Hadamard)

Usage:
    python scripts_qwen35moe/quantize_moe.py \\
        --model /data01/nfs_shared/Qwen3.5-35B-A3B \\
        --mode rotate-full --output_dir ./output/qwen35moe-mode3

    python scripts_qwen35moe/quantize_moe.py \\
        --model /data01/nfs_shared/Qwen3.5-35B-A3B \\
        --mode llm-only --output_dir ./output/qwen35moe-mode1
"""

import argparse
import math
import importlib.util
import os
import re
import sys
import time

# Set CUBLAS deterministic config BEFORE importing torch
if "--deterministic" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

# Force MoE fallback on GPUs without SM90 (A100 has SM80, grouped_mm needs SM90)
def _patch_grouped_mm_for_sm80():
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 9:
                import transformers.integrations.moe as _moe
                _moe._can_use_grouped_mm = lambda *a, **kw: False
                print(f"[Patch] Forced MoE fallback for SM{cc[0]}{cc[1]}")
    except Exception:
        pass

_patch_grouped_mm_for_sm80()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-MoE AutoRound quantization (4 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (Qwen3.5-MoE)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["llm-only", "rotate-merger", "rotate-full", "rotate-quant-all"],
                        help="Quantization mode")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--device_map",
        type=str,
        default="0",
        help='AutoRound calibration device map, e.g. "0", "0,1", "auto", or "balanced"',
    )

    # LLM quantization params
    parser.add_argument("--llm_bits", type=int, default=4)
    parser.add_argument("--llm_group_size", type=int, default=64)
    parser.add_argument("--data_type", type=str, default="int_sym_gptq",
                        help="AutoRound weight data type (default: positive-scale GPTQ symmetric int)")
    parser.add_argument("--attn_bits", type=int, default=None,
                        help="Override all quantized LLM attention linears to this bit-width")
    parser.add_argument("--shared_expert_bits", type=int, default=None,
                        help="Override shared expert gate/up/down projections to this bit-width")
    parser.add_argument("--expert_bits", type=int, default=None,
                        help="Default bit-width for non-shared expert gate/up/down projections")
    parser.add_argument("--expert_up_gate_bits", type=int, default=None,
                        help="Override non-shared expert gate_proj/up_proj bit-width")
    parser.add_argument("--expert_down_bits", type=int, default=None,
                        help="Override non-shared expert down_proj bit-width")
    # Vision quantization params (mode 4 only)
    parser.add_argument("--vision_bits", type=int, default=8)
    parser.add_argument("--vision_group_size", type=int, default=64)

    symmetry = parser.add_mutually_exclusive_group()
    symmetry.add_argument("--sym", dest="sym", action="store_true",
                          help="Use symmetric quantization (default)")
    symmetry.add_argument("--asym", dest="sym", action="store_false",
                          help="Use asymmetric quantization; requires a compatible --data_type")
    parser.set_defaults(sym=True)
    parser.add_argument("--iters", type=int, default=200, help="SignSGD iterations (0=RTN)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulate_steps", type=int, default=1)
    parser.add_argument("--low_gpu_mem_usage", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", type=str, default="auto_gptq",
                        help="Export format: auto_gptq, auto_round, etc.")
    parser.add_argument("--save_rotated", type=str, default=None,
                        help="Save rotated (pre-quant) model for debugging")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable deterministic CuBLAS and attention")
    parser.add_argument("--two_stage", action="store_true", default=False,
                        help="(Mode 4) Two-stage calibration")

    args = parser.parse_args()
    if args.data_type == "int_sym_gptq" and not args.sym:
        parser.error(
            "--data_type int_sym_gptq requires symmetric quantization; remove --asym or use --data_type int"
        )
    return args


def _normalize_cuda_device(token: str) -> str:
    token = str(token).strip()
    if token.isdigit():
        return f"cuda:{token}"
    return token


def _resolve_balanced_spec(spec: str) -> tuple[str, list[str]]:
    """Parse balanced device-map spec into mode + normalized CUDA devices."""
    if not torch.cuda.is_available():
        raise RuntimeError("--device_map balanced requires CUDA")

    mode = "auto"
    raw_devices = None
    lower = spec.lower().strip()

    if lower != "balanced":
        suffix = spec.split(":", 1)[1]
        head, sep, tail = suffix.partition(":")
        normalized_head = head.strip().lower()
        if normalized_head in {"auto", "block"}:
            mode = "auto"
            raw_devices = tail if sep else None
        elif normalized_head in {"manual", "model"}:
            mode = "manual"
            raw_devices = tail if sep else None
        else:
            raw_devices = suffix

    if raw_devices:
        devices = [_normalize_cuda_device(token) for token in raw_devices.split(",") if token.strip()]
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    gpu_devices = [dev for dev in devices if dev.startswith("cuda")]
    if len(gpu_devices) < 2:
        raise ValueError(f"--device_map {spec} requires at least 2 CUDA devices, got: {devices}")

    return mode, gpu_devices


def _expand_layer_pattern(layer_ids):
    return "(?:" + "|".join(str(i) for i in layer_ids) + ")"


def resolve_device_map(device_map: str, model) -> tuple[object, str]:
    """Resolve CLI device_map into an AutoRound-ready device map."""
    if not isinstance(device_map, str):
        return device_map, f"custom:{type(device_map).__name__}"

    spec = device_map.strip()
    lower = spec.lower()
    if lower != "balanced" and not lower.startswith("balanced:"):
        return spec, spec

    mode, gpu_devices = _resolve_balanced_spec(spec)

    if mode == "auto":
        visible_indices = [dev.split(":", 1)[1] for dev in gpu_devices]
        summary = (
            f"balanced block-wise split via AutoRound auto mapper on {', '.join(gpu_devices)} "
            f"(device_map={','.join(visible_indices)})"
        )
        return ",".join(visible_indices), summary

    text_config = getattr(model.config, "text_config", model.config)
    num_layers = text_config.num_hidden_layers
    if not num_layers or num_layers < len(gpu_devices):
        raise ValueError(f"Cannot balance {num_layers} layers across {len(gpu_devices)} GPUs")

    resolved = {
        r"^model\.language_model\.embed_tokens(?:\.|$)": gpu_devices[0],
        r"^model\.language_model\.norm(?:\.|$)": gpu_devices[-1],
        r"^lm_head(?:\.|$)": gpu_devices[-1],
    }

    chunk = math.ceil(num_layers / len(gpu_devices))
    assignments = []
    for idx, device in enumerate(gpu_devices):
        start = idx * chunk
        end = min(num_layers, start + chunk) - 1
        if start >= num_layers:
            break
        layer_ids = list(range(start, end + 1))
        resolved[rf"^model\.language_model\.layers\.{_expand_layer_pattern(layer_ids)}\."] = device
        assignments.append(f"layers[{start}-{end}]→{device}")

    summary = (
        f"balanced manual split on {', '.join(gpu_devices)}; "
        f"embed_tokens→{gpu_devices[0]}, "
        + ", ".join(assignments)
        + f", norm/lm_head→{gpu_devices[-1]}"
    )
    return resolved, summary


ATTN_LINEAR_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_b",
    ".linear_attn.in_proj_a",
    ".linear_attn.out_proj",
)

SHARED_EXPERT_SUFFIXES = (
    ".mlp.shared_expert.gate_proj",
    ".mlp.shared_expert.up_proj",
    ".mlp.shared_expert.down_proj",
)

EXPERT_UP_GATE_SUFFIXES = (
    ".gate_proj",
    ".up_proj",
)

EXPERT_GATE_UP_PATTERN = "model.language_model.layers.*.mlp.experts.*.(gate_proj|up_proj)"
EXPERT_DOWN_PATTERN = "model.language_model.layers.*.mlp.experts.*.down_proj"
_find_spec = importlib.util.find_spec


def _override_cfg(bits: int, group_size: int) -> dict:
    return {"bits": bits, "group_size": group_size}


def validate_export_requirements(format_name: str) -> None:
    formats = [part.strip().lower() for part in format_name.split(",") if part.strip()]
    if any(part == "auto_gptq" for part in formats) and _find_spec("auto_gptq") is None:
        raise RuntimeError(
            "Export format requires the `auto_gptq` package, but it is not installed. "
            "Install `auto-gptq` first or rerun with `--format auto_round:gptqmodel`."
        )


def _uses_pattern(name: str) -> bool:
    return any(token in name for token in ("*", "|", "(", ")", "^", "$"))


def _add_packed_expert_overrides(
    layer_config: dict,
    llm_group_size: int,
    expert_bits: int | None,
    expert_up_gate_bits: int | None,
    expert_down_bits: int | None,
) -> None:
    gate_up_bits = expert_up_gate_bits if expert_up_gate_bits is not None else expert_bits
    down_bits = expert_down_bits if expert_down_bits is not None else expert_bits

    if gate_up_bits is not None:
        layer_config[EXPERT_GATE_UP_PATTERN] = _override_cfg(gate_up_bits, llm_group_size)
    if down_bits is not None:
        layer_config[EXPERT_DOWN_PATTERN] = _override_cfg(down_bits, llm_group_size)


def build_qwen35moe_layer_config(
    model,
    llm_group_size: int,
    attn_bits: int | None = None,
    shared_expert_bits: int | None = None,
    expert_bits: int | None = None,
    expert_up_gate_bits: int | None = None,
    expert_down_bits: int | None = None,
) -> dict:
    """Build explicit per-module bit overrides for Qwen3.5-MoE LLM modules."""
    if all(v is None for v in (attn_bits, shared_expert_bits, expert_bits, expert_up_gate_bits, expert_down_bits)):
        return {}

    layer_config = {}
    saw_expert_linear = False
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if attn_bits is not None and name.endswith(ATTN_LINEAR_SUFFIXES):
            layer_config[name] = _override_cfg(attn_bits, llm_group_size)
            continue

        if shared_expert_bits is not None and name.endswith(SHARED_EXPERT_SUFFIXES):
            layer_config[name] = _override_cfg(shared_expert_bits, llm_group_size)
            continue

        if ".mlp.experts." not in name:
            continue
        saw_expert_linear = True

        if expert_down_bits is not None and name.endswith(".down_proj"):
            layer_config[name] = _override_cfg(expert_down_bits, llm_group_size)
            continue

        if expert_up_gate_bits is not None and name.endswith(EXPERT_UP_GATE_SUFFIXES):
            layer_config[name] = _override_cfg(expert_up_gate_bits, llm_group_size)
            continue

        if expert_bits is not None and name.endswith((".gate_proj", ".up_proj", ".down_proj")):
            layer_config[name] = _override_cfg(expert_bits, llm_group_size)

    if not saw_expert_linear and any(v is not None for v in (expert_bits, expert_up_gate_bits, expert_down_bits)):
        _add_packed_expert_overrides(
            layer_config,
            llm_group_size=llm_group_size,
            expert_bits=expert_bits,
            expert_up_gate_bits=expert_up_gate_bits,
            expert_down_bits=expert_down_bits,
        )

    return layer_config


def _print_llm_override_summary(layer_config: dict) -> None:
    if not layer_config:
        return

    counts = {
        "attention": {"modules": 0, "patterns": 0},
        "shared_expert": {"modules": 0, "patterns": 0},
        "experts": {"modules": 0, "patterns": 0},
    }
    for name in layer_config:
        bucket = "modules"
        if name.endswith(ATTN_LINEAR_SUFFIXES):
            if _uses_pattern(name):
                bucket = "patterns"
            counts["attention"][bucket] += 1
        elif any(token in name for token in SHARED_EXPERT_SUFFIXES):
            if _uses_pattern(name):
                bucket = "patterns"
            counts["shared_expert"][bucket] += 1
        elif ".mlp.experts." in name:
            if _uses_pattern(name):
                bucket = "patterns"
            counts["experts"][bucket] += 1

    print("\nLLM mixed-bit overrides:")
    for key, value in counts.items():
        parts = []
        if value["modules"]:
            parts.append(f"{value['modules']} modules")
        if value["patterns"]:
            parts.append(f"{value['patterns']} regex patterns")
        print(f"  {key}: {', '.join(parts) if parts else '0 modules'}")


def _merge_layer_configs(*configs: dict | None) -> dict | None:
    merged = {}
    for cfg in configs:
        if cfg:
            merged.update(cfg)
    return merged or None


_DUPLICATE_LANGUAGE_MODEL_PREFIX_RE = re.compile(
    r"^model\.language_model\.(?:language_model\.)+"
)


def _normalize_checkpoint_key(key: str) -> str:
    return _DUPLICATE_LANGUAGE_MODEL_PREFIX_RE.sub("model.language_model.", key)


# ─── Post-save checkpoint fix ────────────────────────────────────────────────
#
# When auto_round saves a quantized VLM, two things can go wrong:
#   1. model.save_pretrained() may produce keys with an extra prefix
#      (e.g. model.language_model.language_model.language_model.X)
#   2. copy_missing_tensors_from_source() copies tensors from the ORIGINAL
#      model (not the rotated one) into model_extra_tensors.safetensors
#
# The model shards contain the correct (fused/rotated) weights, but the
# index may point to extra_tensors for non-quantized weights (norms,
# embeddings).  This function fixes the index to prefer shard weights
# and removes duplicates from extra_tensors.
# ─────────────────────────────────────────────────────────────────────────────

def fix_checkpoint_after_save(output_dir: str) -> None:
    """Fix weight_map index to prefer model shards over extra_tensors."""
    import json
    import glob as _glob
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("[fix_checkpoint] safetensors not available, skipping")
        return

    idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    extra_path = os.path.join(output_dir, "model_extra_tensors.safetensors")
    if not os.path.exists(idx_path):
        return

    def _rewrite_safetensors_keys(path: str) -> tuple[dict[str, str], int]:
        if not os.path.exists(path):
            return {}, 0

        renamed = {}
        tensors = {}
        renamed_count = 0
        with safe_open(path, framework="pt", device="cpu") as st:
            for key in st.keys():
                normalized = _normalize_checkpoint_key(key)
                if normalized != key:
                    renamed[key] = normalized
                    renamed_count += 1
                if normalized not in tensors:
                    tensors[normalized] = st.get_tensor(key)

        if renamed_count > 0:
            save_file(tensors, path + ".tmp")
            os.replace(path + ".tmp", path)
        return renamed, renamed_count

    total_renamed = 0
    for shard_path in sorted(_glob.glob(os.path.join(output_dir, "model-*.safetensors"))):
        _, renamed = _rewrite_safetensors_keys(shard_path)
        total_renamed += renamed
    _, renamed = _rewrite_safetensors_keys(extra_path)
    total_renamed += renamed

    # Collect all keys present in model shards (not extra_tensors)
    shard_keys = {}  # key -> shard_filename
    for shard_path in sorted(_glob.glob(os.path.join(output_dir, "model-*.safetensors"))):
        shard_name = os.path.basename(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as st:
            for k in st.keys():
                shard_keys[k] = shard_name

    if not shard_keys:
        return

    # Load index and fix: prefer shard for keys that exist in both
    with open(idx_path) as f:
        idx = json.load(f)
    normalized_weight_map = {}
    for key, value in idx["weight_map"].items():
        normalized_weight_map[_normalize_checkpoint_key(key)] = value
    idx["weight_map"] = normalized_weight_map

    conflicts = 0
    for key in list(idx["weight_map"].keys()):
        if key in shard_keys and idx["weight_map"][key] == "model_extra_tensors.safetensors":
            idx["weight_map"][key] = shard_keys[key]
            conflicts += 1
    # Also add shard keys missing from the index
    for key, shard_name in shard_keys.items():
        if key not in idx["weight_map"]:
            idx["weight_map"][key] = shard_name
            conflicts += 1

    if conflicts == 0 and total_renamed == 0:
        print("[fix_checkpoint] No conflicts found, checkpoint is clean")
        return

    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(
        f"[fix_checkpoint] Normalized {total_renamed} keys and fixed {conflicts} key mappings in index"
    )

    # Remove duplicate keys from extra_tensors
    new_tensors = {}
    removed = 0
    if os.path.exists(extra_path):
        with safe_open(extra_path, framework="pt", device="cpu") as st:
            for k in st.keys():
                if k in shard_keys:
                    removed += 1
                else:
                    new_tensors[k] = st.get_tensor(k)

        if removed > 0:
            save_file(new_tensors, extra_path + ".tmp")
            os.replace(extra_path + ".tmp", extra_path)
            print(f"[fix_checkpoint] Removed {removed} duplicates from extra_tensors "
                  f"({os.path.getsize(extra_path) / 1e9:.2f} GB)")


# ─── Two-stage quantization (Mode 4) ────────────────────────────────────────

def _run_two_stage(model, tokenizer, processor, args, ar_kwargs):
    """Two-stage Mode 4: LLM with pile-10k, vision with llava_conv_58k."""
    import json
    import shutil
    import gc
    from auto_round import AutoRound
    from scripts_qwen35moe.rotation_utils_moe import (
        get_vision_layer_names,
        _get_hadamard_matrix,
        fuse_llm_norms_moe,
        fuse_vision_norms,
        rotate_vision_full,
        rotate_merger_output,
        rotate_llm_moe,
        _get_visual,
    )

    vis_names = get_vision_layer_names(model)
    print(f"\n{'='*60}")
    print(f"TWO-STAGE QUANTIZATION (MoE)")
    print(f"  Stage 1: LLM W{args.llm_bits}G{args.llm_group_size} with pile-10k")
    print(f"  Stage 2: Vision W{args.vision_bits}G{args.vision_group_size} "
          f"with llava_conv_58k ({len(vis_names)} layers)")
    print(f"{'='*60}")

    stage1_dir = os.path.join(args.output_dir, "_stage1_llm")
    stage2_dir = os.path.join(args.output_dir, "_stage2_vision")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    # ── Stage 1: LLM quantization ──
    print(f"\n{'─'*60}")
    print(f"STAGE 1: LLM quantization (W{args.llm_bits}G{args.llm_group_size}, pile-10k)")
    print(f"{'─'*60}")

    ar1 = AutoRound(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        bits=args.llm_bits,
        group_size=args.llm_group_size,
        sym=args.sym,
        data_type=args.data_type,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        dataset="NeelNanda/pile-10k",
        seed=args.seed,
        layer_config=_merge_layer_configs(build_qwen35moe_layer_config(
            model,
            llm_group_size=args.llm_group_size,
            attn_bits=args.attn_bits,
            shared_expert_bits=args.shared_expert_bits,
            expert_bits=args.expert_bits,
            expert_up_gate_bits=args.expert_up_gate_bits,
            expert_down_bits=args.expert_down_bits,
        )),
        quant_nontext_module=False,
        **ar_kwargs,
    )

    t1 = time.time()
    ar1.quantize()
    print(f"Stage 1 quantization: {time.time() - t1:.1f}s")

    print(f"Saving Stage 1 to: {stage1_dir}")
    ar1.save_quantized(stage1_dir, format=args.format)
    fix_checkpoint_after_save(stage1_dir)
    del ar1, model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 2: Reload, re-rotate, quantize vision ──
    print(f"\n{'─'*60}")
    print(f"STAGE 2: Vision quantization (W{args.vision_bits}G{args.vision_group_size}, llava_conv_58k)")
    print(f"{'─'*60}")

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    if args.deterministic:
        model_kwargs["attn_implementation"] = "eager"

    try:
        from transformers import AutoModelForImageTextToText
        model2 = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel
        model2 = AutoModel.from_pretrained(args.model, **model_kwargs)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    text_config = getattr(model2.config, 'text_config', model2.config)
    hidden_size = text_config.hidden_size
    Q_llm = _get_hadamard_matrix(hidden_size, args.device)

    fuse_llm_norms_moe(model2, args.device)
    fuse_vision_norms(model2, args.device)

    vis_config = getattr(model2.config, 'vision_config', None)
    vis_hidden = vis_config.hidden_size if vis_config else _get_visual(model2).blocks[0].attn.qkv.in_features
    Q_vision = _get_hadamard_matrix(vis_hidden, args.device)

    rotate_vision_full(model2, Q_vision, device=args.device)
    rotate_merger_output(model2, Q_llm, device=args.device)
    rotate_llm_moe(model2, Q_llm, args.device)

    vis_names2 = get_vision_layer_names(model2)
    vis_layer_config = {
        name: {"bits": args.vision_bits, "group_size": args.vision_group_size}
        for name in vis_names2
    }

    ar2 = AutoRound(
        model=model2,
        tokenizer=tokenizer,
        processor=processor,
        bits=args.vision_bits,
        group_size=args.vision_group_size,
        sym=args.sym,
        data_type=args.data_type,
        iters=args.iters,
        seqlen=512,
        nsamples=args.nsamples,
        batch_size=1,
        gradient_accumulate_steps=8,
        dataset="liuhaotian/llava_conv_58k",
        seed=args.seed,
        layer_config=vis_layer_config,
        quant_nontext_module=True,
        ignore_layers="model.language_model",
        **ar_kwargs,
    )

    t2 = time.time()
    ar2.quantize()
    print(f"Stage 2 quantization: {time.time() - t2:.1f}s")

    print(f"Saving Stage 2 to: {stage2_dir}")
    ar2.save_quantized(stage2_dir, format=args.format)
    fix_checkpoint_after_save(stage2_dir)
    del ar2, model2
    gc.collect()
    torch.cuda.empty_cache()

    # ── Merge ──
    print(f"\n{'─'*60}")
    print(f"MERGING Stage 1 (LLM) + Stage 2 (Vision)")
    print(f"{'─'*60}")
    _merge_two_stages(stage1_dir, stage2_dir, args.output_dir)
    fix_checkpoint_after_save(args.output_dir)
    print(f"\n✅ Two-stage merge complete → {args.output_dir}")


def _merge_two_stages(stage1_dir, stage2_dir, output_dir):
    """Merge Stage 1 (LLM quantized) + Stage 2 (vision quantized) safetensors."""
    import json
    import shutil
    from safetensors.torch import load_file as st_load, save_file as st_save

    os.makedirs(output_dir, exist_ok=True)

    idx1_path = os.path.join(stage1_dir, "model.safetensors.index.json")
    idx2_path = os.path.join(stage2_dir, "model.safetensors.index.json")

    with open(idx1_path) as f:
        idx1 = json.load(f)
    with open(idx2_path) as f:
        idx2 = json.load(f)

    wmap1 = idx1["weight_map"]
    wmap2 = idx2["weight_map"]

    all_tensors_s1 = {}
    for shard in set(wmap1.values()):
        all_tensors_s1.update(st_load(os.path.join(stage1_dir, shard), device="cpu"))

    all_tensors_s2 = {}
    for shard in set(wmap2.values()):
        all_tensors_s2.update(st_load(os.path.join(stage2_dir, shard), device="cpu"))

    merged = dict(all_tensors_s1)
    vision_from_s2 = 0
    for name, tensor in all_tensors_s2.items():
        if "visual" in name:
            merged[name] = tensor
            vision_from_s2 += 1

    print(f"  Merged: {len(all_tensors_s1)} from Stage 1, {vision_from_s2} vision from Stage 2")
    print(f"  Total: {len(merged)}")

    MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024
    sorted_names = sorted(merged.keys())
    shards = []
    current_shard = {}
    current_size = 0

    for name in sorted_names:
        tensor = merged[name]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    new_weight_map = {}
    total_size = 0
    for i, shard_data in enumerate(shards):
        shard_name = "model.safetensors" if len(shards) == 1 else f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        st_save(shard_data, os.path.join(output_dir, shard_name))
        for name, tensor in shard_data.items():
            new_weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        print(f"  Saved shard: {shard_name} ({len(shard_data)} tensors)")

    if len(shards) > 1:
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, f, indent=2)

    # Merge quantization configs
    qcfg1_path = os.path.join(stage1_dir, "quantize_config.json")
    qcfg2_path = os.path.join(stage2_dir, "quantize_config.json")
    if os.path.exists(qcfg1_path) and os.path.exists(qcfg2_path):
        with open(qcfg1_path) as f:
            qcfg1 = json.load(f)
        with open(qcfg2_path) as f:
            qcfg2 = json.load(f)
        merged_qcfg = qcfg1.copy()
        merged_qcfg["two_stage"] = True
        merged_qcfg["stage1_dataset"] = "NeelNanda/pile-10k"
        merged_qcfg["stage2_dataset"] = "liuhaotian/llava_conv_58k"
        with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
            json.dump(merged_qcfg, f, indent=2)

    # Copy non-safetensor files from Stage 1
    for fname in os.listdir(stage1_dir):
        if fname.endswith(".safetensors") or fname in ("model.safetensors.index.json", "quantize_config.json"):
            continue
        src = os.path.join(stage1_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    validate_export_requirements(args.format)
    total_start = time.time()

    need_rotation = args.mode in ("rotate-merger", "rotate-full", "rotate-quant-all")
    need_full_vision_rotation = args.mode in ("rotate-full", "rotate-quant-all")
    need_vision_quant = args.mode == "rotate-quant-all"

    if args.two_stage and args.mode != "rotate-quant-all":
        print("ERROR: --two_stage only works with --mode rotate-quant-all")
        sys.exit(1)

    # ── Deterministic mode ──
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[deterministic] CuBLAS workspace config set, SDP disabled, cuDNN deterministic")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"LLM: W{args.llm_bits}G{args.llm_group_size}")
    if need_vision_quant:
        print(f"Vision: W{args.vision_bits}G{args.vision_group_size}")
    if args.two_stage:
        print(f"Two-stage: LLM with pile-10k, vision with llava_conv_58k")
    if args.deterministic:
        print(f"Deterministic: ON")
    print(f"Format: {args.format}")
    print(f"Iters: {args.iters}, nsamples: {args.nsamples}, seqlen: {args.seqlen}")
    print(f"AutoRound device_map: {args.device_map}")
    print(f"Batch size: {args.batch_size}, grad_accum: {args.gradient_accumulate_steps}")
    print(f"Low GPU memory mode: {'ON' if args.low_gpu_mem_usage else 'OFF'}")
    print(f"{'='*60}")

    # ── Load model ──
    from transformers import AutoTokenizer, AutoProcessor
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        pass

    load_device = "cpu"
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map=load_device, trust_remote_code=True)
    if args.deterministic:
        model_kwargs["attn_implementation"] = "eager"

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    print(f"Loaded: {type(model).__name__}")

    resolved_device_map, device_map_summary = resolve_device_map(args.device_map, model)
    print(f"Resolved AutoRound device_map: {device_map_summary}")
    if isinstance(resolved_device_map, dict) and not args.low_gpu_mem_usage:
        print("[warn] Manual device_map works best with --low_gpu_mem_usage to avoid cache-stage GPU skew")

    # ── Rotation ──
    if need_rotation:
        from scripts_qwen35moe.rotation_utils_moe import (
            _get_hadamard_matrix,
            fuse_llm_norms_moe,
            fuse_vision_norms,
            rotate_vision_full,
            rotate_merger_output,
            rotate_llm_moe,
            _get_visual,
            get_vision_layer_names,
        )

        print(f"\n{'='*60}")
        print("Applying MoE rotation...")
        print(f"{'='*60}")
        t0 = time.time()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        text_config = getattr(model.config, 'text_config', model.config)
        hidden_size = text_config.hidden_size
        Q_llm = _get_hadamard_matrix(hidden_size, args.device)
        print(f"  Q_llm: {Q_llm.shape} (hidden_size={hidden_size}, exact_hadamard={hidden_size & (hidden_size - 1) == 0})")

        # Step 1: Fuse LLM norms (MoE version)
        print("  Fusing LLM norms (MoE)...")
        fuse_llm_norms_moe(model, args.device)

        # Step 2: Full vision rotation (modes 3, 4)
        if need_full_vision_rotation:
            print("  Fusing vision norms...")
            fuse_vision_norms(model, args.device)

            vis_config = getattr(model.config, 'vision_config', None)
            vis_hidden = vis_config.hidden_size if vis_config else _get_visual(model).blocks[0].attn.qkv.in_features
            Q_vision = _get_hadamard_matrix(vis_hidden, args.device)
            print(f"  Q_vision: {Q_vision.shape} (vis_hidden={vis_hidden})")

            print("  Rotating full vision encoder...")
            rotate_vision_full(model, Q_vision, device=args.device)

        # Step 3: Rotate merger output → LLM space
        if need_rotation:
            try:
                print("  Rotating merger output → LLM space...")
                rotate_merger_output(model, Q_llm, device=args.device)
            except AttributeError:
                print("  No visual module found, skipping merger rotation.")

        # Step 4: Rotate MoE LLM
        print("  Rotating MoE LLM layers...")
        rotate_llm_moe(model, Q_llm, args.device)

        rot_time = time.time() - t0
        print(f"Rotation completed in {rot_time:.1f}s")

        if args.save_rotated:
            print(f"Saving rotated model to: {args.save_rotated}")
            model.save_pretrained(args.save_rotated, max_shard_size="5GB")
            tokenizer.save_pretrained(args.save_rotated)

    # ── Build layer_config for mode 4 ──
    llm_layer_config = build_qwen35moe_layer_config(
        model,
        llm_group_size=args.llm_group_size,
        attn_bits=args.attn_bits,
        shared_expert_bits=args.shared_expert_bits,
        expert_bits=args.expert_bits,
        expert_up_gate_bits=args.expert_up_gate_bits,
        expert_down_bits=args.expert_down_bits,
    )
    _print_llm_override_summary(llm_layer_config)

    layer_config = llm_layer_config or None
    quant_nontext = False
    if need_vision_quant and not args.two_stage:
        from scripts_qwen35moe.rotation_utils_moe import get_vision_layer_names
        quant_nontext = True
        vis_names = get_vision_layer_names(model)
        vision_layer_config = {
            name: {"bits": args.vision_bits, "group_size": args.vision_group_size}
            for name in vis_names
        }
        layer_config = _merge_layer_configs(llm_layer_config, vision_layer_config)
        print(f"\nVision quantization: {len(vis_names)} layers at W{args.vision_bits}G{args.vision_group_size}")

    from auto_round import AutoRound

    ar_kwargs = {}
    if args.deterministic:
        ar_kwargs["enable_deterministic_algorithms"] = True
    ar_kwargs["device_map"] = resolved_device_map
    ar_kwargs["low_gpu_mem_usage"] = args.low_gpu_mem_usage

    if args.two_stage and need_vision_quant:
        _run_two_stage(model, tokenizer, processor, args, ar_kwargs)
    else:
        # ── Single-stage quantization ──
        print(f"\n{'='*60}")
        print(f"AutoRound quantization: LLM W{args.llm_bits}G{args.llm_group_size}")
        print(f"Dataset: {args.dataset}, iters={args.iters}, nsamples={args.nsamples}")
        print(f"{'='*60}")

        ar = AutoRound(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            bits=args.llm_bits,
            group_size=args.llm_group_size,
            sym=args.sym,
            data_type=args.data_type,
            iters=args.iters,
            seqlen=args.seqlen,
            nsamples=args.nsamples,
            batch_size=args.batch_size,
            gradient_accumulate_steps=args.gradient_accumulate_steps,
            dataset=args.dataset,
            seed=args.seed,
            layer_config=layer_config,
            quant_nontext_module=quant_nontext,
            **ar_kwargs,
        )

        t1 = time.time()
        ar.quantize()
        quant_time = time.time() - t1
        print(f"Quantization completed in {quant_time:.1f}s ({quant_time / 3600:.2f}h)")

        print(f"\nSaving to: {args.output_dir} (format={args.format})")
        ar.save_quantized(args.output_dir, format=args.format)
        fix_checkpoint_after_save(args.output_dir)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ Done! Mode={args.mode}")
    print(f"   Output: {args.output_dir}")
    print(f"   LLM: W{args.llm_bits}G{args.llm_group_size}")
    if need_vision_quant:
        print(f"   Vision: W{args.vision_bits}G{args.vision_group_size}")
    if args.two_stage:
        print(f"   Strategy: two-stage (LLM=pile-10k, vision=llava_conv_58k)")
    print(f"   Time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
