# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public GLM-5.3-Flash W8 / routed W4-G64 producer.

Native block-FP8 is streamed to a BF16 calibration checkpoint. GPTQ optimizes
decoder projections and all routed/shared experts. Vision, head, router,
indexer, mHC and the unabsorbed DSA K/V factors remain floating point in this
checkpoint for the deployment W8 quantizer. KDA decay/beta remain floating
point at both boundaries. No transformed/absorbed weight is quantized twice.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import random
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
EXPERT_PATTERN = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _weight_map(path):
    index = path / "model.safetensors.index.json"
    if index.is_file():
        return _json(index)["weight_map"]
    with safe_open(path / "model.safetensors", framework="pt") as reader:
        return dict.fromkeys(reader.keys(), "model.safetensors")


def _source_identity(path, layers):
    config = (path / "config.json").read_bytes()
    index = path / "model.safetensors.index.json"
    return {
        "source_model_dir": str(path),
        "num_layers": layers,
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()
        if index.exists()
        else None,
    }


def _prepare_shard(args):
    source, target, filename, keys, weight_map, block, device = args
    torch.set_num_threads(1)
    result = {}
    with ExitStack() as stack:
        readers = {}

        def read(key):
            shard = weight_map[key]
            if shard not in readers:
                readers[shard] = stack.enter_context(
                    safe_open(Path(source) / shard, framework="pt")
                )
            return readers[shard].get_tensor(key)

        for key in keys:
            if key.endswith("_scale_inv"):
                continue
            value = read(key)
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                scales = read(key + "_scale_inv").to(device=device, dtype=torch.float32)
                br, bc = block
                if value.ndim != 2:
                    raise ValueError(f"FP8 matrix required: {key}")
                rows, cols = value.shape
                if tuple(scales.shape) != (
                    (rows + br - 1) // br,
                    (cols + bc - 1) // bc,
                ):
                    raise ValueError(f"Invalid FP8 scale shape for {key}")
                if not torch.isfinite(scales).all() or (scales <= 0).any():
                    raise ValueError(f"Invalid FP8 scales for {key}")
                value = value.to(device).float()
                if rows % br == 0 and cols % bc == 0:
                    value = (
                        value.reshape(rows // br, br, cols // bc, bc)
                        * scales[:, None, :, None]
                    ).reshape(rows, cols)
                else:
                    value *= scales.repeat_interleave(br, 0).repeat_interleave(bc, 1)[
                        :rows, :cols
                    ]
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite dequantized weight: {key}")
                value = value.to(device="cpu", dtype=torch.bfloat16)
            # Preserve native FP32 mHC/router/norm tensors, not just BF16 weights.
            result[key] = value.contiguous()
    if result:
        save_file(result, str(Path(target) / filename))
    return (
        filename,
        list(result),
        sum(t.numel() * t.element_size() for t in result.values()),
    )


def prepare_glm5_3_flash_bf16(
    *,
    model_dir: str,
    output_dir: str,
    num_layers: int | None = None,
    devices: tuple[str, ...] = ("cpu",),
) -> str:
    """Bound memory by source shards and publish only a complete checkpoint."""
    source, output = Path(model_dir).resolve(), Path(output_dir).resolve()
    config = _json(source / "config.json")
    if config.get("model_type") != "glm5_next":
        raise ValueError("Expected the full multimodal glm5_next checkpoint")
    text = config["text_config"]
    layers = int(text["num_hidden_layers"] if num_layers is None else num_layers)
    if not 1 <= layers <= text["num_hidden_layers"] or not devices:
        raise ValueError("Invalid layer count or empty preparation devices")
    identity = _source_identity(source, layers)
    if output.exists():
        if (output / "glm53_preparation.json").is_file() and _json(
            output / "glm53_preparation.json"
        )["source"] == identity:
            prepared_config = _json(output / "config.json")
            if prepared_config["text_config"].get(
                "num_nextn_predict_layers"
            ) != text.get("num_nextn_predict_layers"):
                raise ValueError(
                    "Prepared checkpoint changed the source's MTP configuration metadata"
                )
            for shard in set(_weight_map(output).values()):
                if not (output / shard).is_file():
                    raise FileNotFoundError(output / shard)
            return str(output)
        raise FileExistsError(
            f"Refusing to overwrite an unrelated/incomplete prepared checkpoint: {output}"
        )
    quant = config.pop("quantization_config", {}) or {}
    if quant.get("quant_method") not in (None, "fp8"):
        raise ValueError("Preparation requires unquantized or native block-FP8 weights")
    weight_map = _weight_map(source)
    groups = {}
    for key, shard in weight_map.items():
        match = _LAYER.match(key)
        if match and int(match[1]) >= layers:
            continue  # Includes the release's layer-45 MTP branch.
        groups.setdefault(shard, []).append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent)
    )
    try:
        jobs = [
            (
                str(source),
                str(staging),
                shard,
                keys,
                weight_map,
                tuple(quant.get("weight_block_size", (128, 128))),
                devices[i % len(devices)],
            )
            for i, (shard, keys) in enumerate(sorted(groups.items()))
        ]
        new_map, total = {}, 0
        if len(devices) == 1:
            results = map(_prepare_shard, jobs)
            for shard, keys, size in results:
                new_map.update(dict.fromkeys(keys, shard))
                total += size
                print(f"GLM preparation: {shard}, {size / 1e9:.2f} GB", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=len(devices),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                for shard, keys, size in pool.map(_prepare_shard, jobs):
                    new_map.update(dict.fromkeys(keys, shard))
                    total += size
                    print(f"GLM preparation: {shard}, {size / 1e9:.2f} GB", flush=True)
        text["num_hidden_layers"] = layers
        # The native Transformers class constructs only num_hidden_layers.
        # Preserve the release's metadata, while omitting the unimplemented
        # MTP tensors above; never change a source identity field for export.
        for key in ("layer_types", "mlp_layer_types", "indexer_types"):
            text[key] = text[key][:layers]
        # Config derives the flattened KDA fields from this release dictionary.
        if "linear_attn_config" in text:
            for key in ("kda_layers", "full_attn_layers"):
                text["linear_attn_config"][key] = [
                    i for i in text["linear_attn_config"].get(key, []) if i < layers
                ]
        _write(staging / "config.json", config)
        _write(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": total}, "weight_map": new_map},
        )
        from gptqmodel.utils.model_dequant import copy_aux_files

        copy_aux_files(
            source,
            staging,
            set(weight_map.values()) | {"config.json", "model.safetensors.index.json"},
        )
        _write(
            staging / "glm53_preparation.json",
            {"source": identity, "total_size": total, "mtp_included": False},
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return str(output)


def expected_glm53_quantized_modules(config):
    """Independent precision/shape inventory, including every routed expert."""
    c = config.get_text_config()
    h, qkv = c.hidden_size, c.linear_num_heads * c.linear_head_dim
    expected = {}
    for i, kind in enumerate(c.layer_types):
        root = f"model.language_model.layers.{i}."
        if kind == "linear_attention":
            projections = {
                "q_proj": (h, qkv),
                "k_proj": (h, qkv),
                "v_proj": (h, qkv),
                "g_a_proj": (h, c.linear_head_dim),
                "g_b_proj": (c.linear_head_dim, qkv),
                "o_proj": (qkv, h),
            }
        else:
            projections = {
                "q_a_proj": (h, c.q_lora_rank),
                "q_b_proj": (c.q_lora_rank, c.num_attention_heads * c.qk_nope_head_dim),
                "kv_a_proj_with_mqa": (h, c.kv_lora_rank),
                "o_proj": (c.num_attention_heads * c.v_head_dim, h),
            }
        expected.update(
            {root + "self_attn." + n: (8, a, b) for n, (a, b) in projections.items()}
        )
        mlps = (
            [("mlp", 8, c.intermediate_size)]
            if c.mlp_layer_types[i] == "dense"
            else [
                ("mlp.shared_experts", 8, c.moe_intermediate_size * c.n_shared_experts),
                *[
                    (f"mlp.experts.{e}", 4, c.moe_intermediate_size)
                    for e in range(c.n_routed_experts)
                ],
            ]
        )
        for name, bits, width in mlps:
            for proj, a, b in (
                ("gate_proj", h, width),
                ("up_proj", h, width),
                ("down_proj", width, h),
            ):
                expected[root + name + "." + proj] = bits, a, b
    return expected


def audit_glm5_3_flash_quantized_artifact(model_dir: str | Path) -> dict:
    from transformers import AutoConfig

    root = Path(model_dir)
    config = AutoConfig.from_pretrained(root)
    quant = config.quantization_config
    if (
        quant.get("bits") != 8
        or quant.get("group_size") != 64
        or quant.get("sym") is not True
        or quant.get("desc_act", False) is not False
    ):
        raise ValueError(
            "Expected symmetric W8 / W4 G64 checkpoint with desc_act=False"
        )
    override = (quant.get("dynamic") or {}).get(EXPERT_PATTERN, {})
    if override != {"bits": 4, "group_size": 64, "sym": True}:
        raise ValueError("Missing routed W4 G64 precision override")
    expected, weights = expected_glm53_quantized_modules(config), _weight_map(root)
    actual = {
        key.removesuffix(".qweight") for key in weights if key.endswith(".qweight")
    }
    if actual != set(expected):
        raise ValueError(
            f"GPTQ inventory mismatch: missing={sorted(set(expected) - actual)[:8]}, extra={sorted(actual - set(expected))[:8]}"
        )
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(safe_open(root / name, framework="pt"))
            for name in set(weights.values())
        }
        if {k for r in readers.values() for k in r.keys()} != set(weights):
            raise ValueError("GPTQ index and shard headers disagree")
        for prefix, (bits, inputs, outputs) in expected.items():
            factor, groups = 32 // bits, (inputs + 63) // 64
            specs = {
                "qweight": ((inputs // factor, outputs), "I32"),
                "qzeros": ((groups, outputs // factor), "I32"),
                "scales": ((groups, outputs), "F16"),
                "g_idx": ((inputs,), "I32"),
            }
            for suffix, (shape, dtype) in specs.items():
                key = prefix + "." + suffix
                item = readers[weights[key]].get_slice(key)
                if tuple(item.get_shape()) != shape or item.get_dtype() != dtype:
                    raise ValueError(
                        f"Invalid W{bits} G64 tensor {key}: {item.get_shape()}, {item.get_dtype()}"
                    )
            key = prefix + ".scales"
            scales = readers[weights[key]].get_tensor(key)
            if not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError(f"Invalid GPTQ scales: {key}")
            key = prefix + ".g_idx"
            groups = readers[weights[key]].get_tensor(key)
            if not torch.equal(groups, torch.arange(inputs, dtype=torch.int32) // 64):
                raise ValueError(f"GPTQ requires consecutive G64 groups: {key}")
    result = {
        "layers": config.text_config.num_hidden_layers,
        "w4_modules": sum(x[0] == 4 for x in expected.values()),
        "w8_modules": sum(x[0] == 8 for x in expected.values()),
        "tensor_count": len(weights),
    }
    return result


@dataclass(frozen=True)
class Glm53QuantRecipeResult:
    output_dir: str
    prepared_model_dir: str
    method: str
    artifact_format: str
    model_family: str
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


def _select_calibration_rows(tokenized):
    """Make GPTQ's existing short-row policy explicit before replay.

    The generic quantize entrypoint currently does not forward its minimum-
    length argument to processors. Use its supported default consistently,
    recording the actual rows and tokens instead of claiming all requested
    rows were calibrated.
    """
    selected = [row for row in tokenized if len(row["input_ids"]) > 10]
    if not selected:
        raise ValueError("GLM calibration has no rows longer than 10 tokens")
    evidence = {
        "minimum_length_exclusive": 10,
        "actual_nsamples": len(selected),
        "dropped_short_samples": len(tokenized) - len(selected),
        "actual_tokens": sum(sum(row["attention_mask"]) for row in selected),
        "tokenized_sha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode()
        ).hexdigest(),
    }
    return selected, evidence


def quantize_glm5_3_flash(
    *,
    model_dir: str,
    output_dir: str,
    prepared_model_dir: str | None = None,
    num_layers: int | None = None,
    method: str = "gptq",
    artifact_format: str = "gptqmodel_hf",
    bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    nsamples: int = 128,
    seqlen: int = 2048,
    batch_size: int = 1,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = "NeelNanda/pile-10k",
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
    device: str = "cuda:0",
    preparation_devices: tuple[str, ...] = ("cpu",),
    moe_routing_batch_size: int = 32,
    auto_forward_data_parallel: bool = False,
    hessian_mse: bool = True,
    wait_for_submodule_finalizers: bool = True,
    seed: int = 42,
    dry_run: bool = False,
) -> Glm53QuantRecipeResult:
    if (method, artifact_format, bits, expert_bits, group_size, sym) != (
        "gptq",
        "gptqmodel_hf",
        8,
        4,
        64,
        True,
    ):
        raise ValueError(
            "GLM-5.3 recipe supports GPTQ symmetric W8 / routed W4 G64 only"
        )
    if min(nsamples, seqlen, batch_size, moe_routing_batch_size) <= 0:
        raise ValueError(
            "Calibration counts, lengths and routing batch size must be positive"
        )
    source, output = Path(model_dir).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Quantized output must be outside the source checkpoint")
    prepared = (
        Path(prepared_model_dir).resolve()
        if prepared_model_dir
        else output.with_name(output.name + "_bf16_input")
    )
    result = Glm53QuantRecipeResult(
        str(output),
        str(prepared),
        method,
        artifact_format,
        "glm5_3_flash",
        {
            "routed_non_shared_experts": "W4 G64",
            "other_gptq_linears": "W8 G64",
            "deployment_rtn_w8": [
                "vision",
                "lm_head",
                "router",
                "indexer",
                "mhc",
                "conv1d",
                "DSA kv_b factors",
            ],
            "floating_kda": ["forget_gate.f_a_proj", "forget_gate.f_b_proj", "b_proj"],
            "activation": "A16 at deployment",
            "mla_layout": "hf_unabsorbed",
        },
        {
            "nsamples": nsamples,
            "seqlen": seqlen,
            "jsonl": calibration_jsonl,
            "dataset": calibration_dataset,
            "text_key": calibration_text_key,
            "seed": seed,
        },
        {
            "recipe": "gptqmodel.recipes.glm5_3_flash.quantize_glm5_3_flash",
            "dry_run": dry_run,
            "source": str(source),
            "num_layers": num_layers,
            "mtp_included": False,
            "moe_routing": "clamp_preserving_bypass",
            "moe_routing_batch_size": moe_routing_batch_size,
            "auto_forward_data_parallel": auto_forward_data_parallel,
            "hessian_mse": hessian_mse,
            "wait_for_submodule_finalizers": wait_for_submodule_finalizers,
        },
    )
    if dry_run:
        return result
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig
    from .ling3_flash import _load_calibration_texts, tokenize_ling3_flash_calibration

    random.seed(seed)
    torch.manual_seed(seed)
    texts = _load_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=Path(calibration_jsonl) if calibration_jsonl else None,
        calibration_dataset=calibration_dataset,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenized = tokenize_ling3_flash_calibration(
        texts, tokenizer=tokenizer, seqlen=seqlen
    )
    if len(tokenized) != nsamples:
        raise ValueError("Calibration sample count does not match requested nsamples")
    tokenized, calibration_evidence = _select_calibration_rows(tokenized)
    result.calibration.update(calibration_evidence)
    prepare_glm5_3_flash_bf16(
        model_dir=str(source),
        output_dir=str(prepared),
        num_layers=num_layers,
        devices=preparation_devices,
    )
    qcfg = QuantizeConfig(
        bits=8,
        dynamic={EXPERT_PATTERN: {"bits": 4, "group_size": 64, "sym": True}},
        group_size=64,
        sym=True,
        desc_act=False,
        device=device,
        moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=moe_routing_batch_size)),
        offload_to_disk=False,
        auto_forward_data_parallel=auto_forward_data_parallel,
        hessian_mse=hessian_mse,
        wait_for_submodule_finalizers=wait_for_submodule_finalizers,
    )
    model = GPTQModel.load(
        str(prepared), quantize_config=qcfg, attn_implementation="eager"
    )
    model.quantize(tokenized, batch_size=batch_size)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.quant-", dir=output.parent))
    try:
        model.save(str(staging))
        tokenizer.save_pretrained(staging)
        from gptqmodel.utils.model_dequant import copy_aux_files

        copy_aux_files(
            source,
            staging,
            set(_weight_map(source).values())
            | {"config.json", "model.safetensors.index.json"},
        )
        result.provenance["audit"] = audit_glm5_3_flash_quantized_artifact(staging)
        _write(staging / "glm53_quantization_recipe.json", asdict(result))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result
