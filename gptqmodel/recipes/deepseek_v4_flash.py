"""Stable W8/W4-G64 GPTQ and AutoRound recipe for DeepSeek-V4-Flash.

The published Flash checkpoint is a mixed native format: most dense weights
are FP8 E4M3 with direct UE8M0 scales, while routed experts are packed E2M1
FP4 with direct E8M0 scales.  GPTQ/AutoRound calibration needs ordinary
floating-point modules, so this recipe first streams only the requested decoder
layers into a BF16 checkpoint. ``num_layers=6`` therefore creates a real
six-layer model rather than loading all 43 layers and merely skipping work.

The deployment profile follows the repository's other MoE recipes: routed
non-shared expert linears are W4, all other registered quantizable linears are
W8, and both use symmetric group-size 64 quantization. Disk offload is opt-in.

DeepSeek-V4 already stores the efficient absorbed latent-attention form
(``q_a/q_b``, shared ``kv``, and grouped ``o_a/o_b``).  The preparation step
preserves that factorization; it never expands a conventional K/V projection
or multiplies ``o_a`` and ``o_b`` together.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import (
    convert_fp8_shard,
    copy_aux_files,
    load_json,
    normalize_device,
    resolve_block_size,
    write_json,
)

_LAYER_KEY = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
_MTP_KEY = re.compile(r"^(?:model\.)?mtp\.")
_ROUTED_EXPERT_PATTERN = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
_SUPPORTED_METHODS = {"gptq", "auto_round"}


@dataclass(frozen=True)
class DeepSeekV4FlashPreparationResult:
    source_model_dir: str
    output_dir: str
    num_layers: int
    tensor_count: int
    shard_count: int
    total_size: int
    target_dtype: str


@dataclass(frozen=True)
class DeepSeekV4FlashQuantRecipeResult:
    source_model_dir: str
    prepared_model_dir: str
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    num_layers: int | None
    offload_to_disk: bool
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DeepSeekV4FlashArtifactAudit:
    """Header-only proof of the packed W8/W4-G64 checkpoint inventory."""

    model_dir: str
    num_layers: int
    tensor_count: int
    shard_count: int
    quantized_module_count: int
    w4_module_count: int
    w8_module_count: int


class _SelectedSafeTensorReader:
    """Expose a shard subset through the reader API used by dequantization."""

    def __init__(self, reader, keys: Sequence[str]):
        self._reader = reader
        self._keys = tuple(keys)

    def keys(self) -> list[str]:
        return list(self._keys)

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        return self._reader.get_tensor(key)


def select_deepseek_v4_flash_checkpoint_keys(
    weight_map: Mapping[str, str],
    *,
    num_layers: int,
    include_mtp: bool = False,
) -> set[str]:
    """Select global tensors and decoder layers ``[0, num_layers)``.

    The native checkpoint uses bare ``layers.N`` keys while a canonical HF
    checkpoint may use ``model.layers.N``.  Both layouts are accepted.  MTP is
    intentionally excluded from the inference model unless explicitly asked
    for by a lower-level caller.
    """

    if int(num_layers) <= 0:
        raise ValueError("num_layers must be positive")

    selected: set[str] = set()
    for key in weight_map:
        layer_match = _LAYER_KEY.match(key)
        if layer_match is not None:
            if int(layer_match.group(1)) < int(num_layers):
                selected.add(key)
            continue
        if _MTP_KEY.match(key):
            if include_mtp:
                selected.add(key)
            continue
        selected.add(key)

    # Keep scale companions even for checkpoint layouts that place them outside
    # a decoder-prefix grouping used by a custom exporter.
    for key in tuple(selected):
        if not key.endswith(".weight"):
            continue
        for scale_key in (
            key + "_scale_inv",
            key[: -len(".weight")] + ".scale",
        ):
            if scale_key in weight_map:
                selected.add(scale_key)
    return selected


def prepare_deepseek_v4_flash_bf16(
    *,
    model_dir: str,
    output_dir: str,
    num_layers: int | None = None,
    device: str | None = "cuda:0",
) -> DeepSeekV4FlashPreparationResult:
    """Stream a native V4 Flash checkpoint into a BF16 GPTQ input model.

    The write is staged in a sibling temporary directory and renamed only
    after every shard, index, config, and tokenizer asset has been produced.
    Existing output paths are never overwritten.
    """

    source_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"DeepSeek-V4 model directory not found: {source_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output_path}")

    config = load_json(source_path / "config.json")
    if config.get("model_type") != "deepseek_v4":
        raise ValueError(
            f"Expected model_type='deepseek_v4', got {config.get('model_type')!r}"
        )
    quantization_config = config.get("quantization_config") or {}
    if str(quantization_config.get("quant_method", "")).lower() != "fp8":
        raise ValueError(
            "Preparation expects the native DeepSeek-V4 FP8/FP4 checkpoint"
        )

    index_path = source_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Sharded safetensors index is required: {index_path}")
    source_index = load_json(index_path)
    weight_map = dict(source_index.get("weight_map") or {})
    if not weight_map:
        raise ValueError(f"weight_map is empty in {index_path}")

    source_layers = int(config.get("num_hidden_layers", 0))
    target_layers = source_layers if num_layers is None else int(num_layers)
    if source_layers <= 0:
        raise ValueError("num_hidden_layers must be positive in source config")
    if not 1 <= target_layers <= source_layers:
        raise ValueError(
            f"num_layers must be in [1, {source_layers}], got {target_layers}"
        )

    selected_keys = select_deepseek_v4_flash_checkpoint_keys(
        weight_map,
        num_layers=target_layers,
        include_mtp=False,
    )
    selected_by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        filename = weight_map[key]
        selected_by_shard.setdefault(filename, []).append(key)

    # convert_fp8_shard resolves a scale through its current reader.  Fail
    # clearly if an unusual checkpoint splits a weight and its companion scale.
    for key in selected_keys:
        if not key.endswith(".weight"):
            continue
        for scale_key in (
            key + "_scale_inv",
            key[: -len(".weight")] + ".scale",
        ):
            if scale_key in selected_keys and weight_map[scale_key] != weight_map[key]:
                raise ValueError(
                    f"Weight and scale must share a shard: {key} vs {scale_key}"
                )

    normalized_device = normalize_device(device)
    if normalized_device is not None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested preparation device")
    open_device = normalized_device or "cpu"
    block_shape = resolve_block_size(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.prepare-",
            dir=str(output_path.parent),
        )
    )
    output_weight_map: dict[str, str] = {}
    total_size = 0
    try:
        for filename in sorted(selected_by_shard):
            source_shard = source_path / filename
            keys = sorted(selected_by_shard[filename])
            with safe_open(
                source_shard,
                framework="pt",
                device=open_device,
            ) as reader:
                selected_reader = _SelectedSafeTensorReader(reader, keys)
                tensors = convert_fp8_shard(
                    selected_reader,
                    torch.bfloat16,
                    block_shape=block_shape,
                    model_type="deepseek_v4",
                )
            if not tensors:
                continue
            save_file(tensors, str(staging_path / filename))
            output_weight_map.update({key: filename for key in tensors})
            total_size += sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
            del tensors
            if normalized_device is not None:
                torch.cuda.empty_cache()

        new_index = dict(source_index)
        metadata = dict(new_index.get("metadata") or {})
        metadata["total_size"] = int(total_size)
        new_index["metadata"] = metadata
        new_index["weight_map"] = output_weight_map
        write_json(staging_path / "model.safetensors.index.json", new_index)

        new_config = _build_prepared_config(
            config,
            num_layers=target_layers,
        )
        write_json(staging_path / "config.json", new_config)

        skip_files = set(weight_map.values()) | {
            "config.json",
            "model.safetensors.index.json",
        }
        copy_aux_files(source_path, staging_path, skip_files)

        from gptqmodel.utils.deepseek_v4_dspark import preserve_deepseek_v4_dspark

        preserve_deepseek_v4_dspark(source_path, staging_path)

        result = DeepSeekV4FlashPreparationResult(
            source_model_dir=str(source_path),
            output_dir=str(output_path),
            num_layers=target_layers,
            tensor_count=len(output_weight_map),
            shard_count=len(set(output_weight_map.values())),
            total_size=int(total_size),
            target_dtype="bfloat16",
        )
        (staging_path / "deepseek_v4_flash_bf16_preparation.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging_path.rename(output_path)
        return result
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def build_deepseek_v4_flash_dynamic(
    *,
    num_layers: int | None = None,
    expert_bits: int = 4,
    group_size: int = 64,
    max_quant_layers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build routed-expert W4 overrides and optional trailing exclusions."""

    dynamic: dict[str, dict[str, Any]] = {
        _ROUTED_EXPERT_PATTERN: {
            "bits": int(expert_bits),
            "group_size": int(group_size),
            "sym": True,
        }
    }
    if max_quant_layers is None:
        return dynamic
    if num_layers is None:
        raise ValueError("num_layers is required when max_quant_layers is set")
    if not 0 <= int(max_quant_layers) <= int(num_layers):
        raise ValueError(f"max_quant_layers must be in [0, {num_layers}]")
    return {
        **{
            rf"-:.*\.layers\.{layer_index}\..*": {}
            for layer_index in range(int(max_quant_layers), int(num_layers))
        },
        **dynamic,
    }


def tokenize_deepseek_v4_flash_calibration(
    texts: list[str],
    *,
    tokenizer,
    seqlen: int,
) -> list[dict[str, list[int]]]:
    """Tokenize and hard-truncate calibration rows."""

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(seqlen),
        padding=False,
    )
    input_ids = encoded["input_ids"]
    attention_masks = encoded.get("attention_mask")
    if attention_masks is None:
        attention_masks = [[1] * len(row) for row in input_ids]
    if len(input_ids) != len(texts) or len(attention_masks) != len(texts):
        raise RuntimeError("Tokenizer returned a different number of calibration rows")

    rows: list[dict[str, list[int]]] = []
    for row_index, (row_ids, row_mask) in enumerate(
        zip(input_ids, attention_masks, strict=True)
    ):
        if len(row_ids) > int(seqlen):
            raise RuntimeError(
                f"Tokenizer did not truncate calibration row {row_index} to {seqlen}"
            )
        if len(row_ids) != len(row_mask):
            raise RuntimeError(
                f"Calibration row {row_index} has mismatched ids and attention mask"
            )
        rows.append({"input_ids": list(row_ids), "attention_mask": list(row_mask)})
    return rows


def quantize_deepseek_v4_flash(
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
    batch_size: int = 1,
    nsamples: int = 128,
    seqlen: int = 2048,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = "NeelNanda/pile-10k",
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
    device: str = "cuda:0",
    preparation_device: str | None = "cuda:0",
    trust_remote_code: bool = True,
    offload_to_disk: bool = False,
    offload_path: str | None = None,
    hessian_mse: bool = True,
    wait_for_submodule_finalizers: bool = True,
    moe_routing_batch_size: int | None = 128,
    auto_forward_data_parallel: bool = False,
    max_quant_layers: int | None = None,
    auto_round_version: str = "v2",
    auto_round_iters: int = 200,
    auto_round_lr: float | None = None,
    auto_round_minmax_lr: float | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> DeepSeekV4FlashQuantRecipeResult:
    """Quantize V4 Flash to symmetric W8/W4-G64 with GPTQ or AutoRound."""

    normalized_method = _normalize_method(method)
    normalized_format = str(artifact_format).lower()
    _validate_quant_recipe_args(
        method=normalized_method,
        artifact_format=normalized_format,
        bits=bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=sym,
        batch_size=batch_size,
        nsamples=nsamples,
        seqlen=seqlen,
        num_layers=num_layers,
        max_quant_layers=max_quant_layers,
        moe_routing_batch_size=moe_routing_batch_size,
        auto_round_version=auto_round_version,
        auto_round_iters=auto_round_iters,
        auto_round_lr=auto_round_lr,
        auto_round_minmax_lr=auto_round_minmax_lr,
    )

    source_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    target_layers = None if num_layers is None else int(num_layers)
    if prepared_model_dir:
        prepared_path = Path(prepared_model_dir).expanduser().resolve()
    else:
        suffix = "full" if target_layers is None else f"first{target_layers}"
        prepared_path = output_path.parent / f"deepseek-v4-flash-bf16-{suffix}"
    calibration_path = (
        None
        if calibration_jsonl in (None, "")
        else Path(str(calibration_jsonl)).expanduser().resolve()
    )
    resolved_offload_path = _resolve_offload_path(
        output_path=output_path,
        offload_to_disk=offload_to_disk,
        offload_path=offload_path,
    )
    dynamic = build_deepseek_v4_flash_dynamic(
        num_layers=target_layers,
        expert_bits=expert_bits,
        group_size=group_size,
        max_quant_layers=max_quant_layers,
    )

    precision = {
        "routed_non_shared_experts": f"W{int(expert_bits)}",
        "shared_expert_and_other_quantized_linears": f"W{int(bits)}",
        "group_size": int(group_size),
        "symmetric": True,
        "grouped_o_a_projection": "BF16",
        "compressor_indexer_router_norm_mhc": (
            "native dtype; official FP32-sensitive tensors retained"
        ),
        "native_input": "FP8-E4M3-B128x128 + routed-E2M1-FP4-B32",
    }
    calibration = {
        "jsonl": None if calibration_path is None else str(calibration_path),
        "dataset": calibration_dataset,
        "text_key": str(calibration_text_key),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
    }
    provenance = {
        "recipe": "gptqmodel.recipes.deepseek_v4_flash.quantize_deepseek_v4_flash",
        "backend": (
            "native_stream_dequant_then_direct_gptqmodel_" f"{normalized_method}_api"
        ),
        "dry_run": bool(dry_run),
        "mla_weight_form": "shared_latent_and_grouped_output_preserved",
        "mtp_included": False,
        "offload_to_disk": bool(offload_to_disk),
        "offload_path": resolved_offload_path,
        "auto_forward_data_parallel": bool(auto_forward_data_parallel),
        "wait_for_submodule_finalizers": bool(wait_for_submodule_finalizers),
        "hessian_mse": bool(hessian_mse),
        "moe_routing": "bypass",
        "moe_routing_batch_size": moe_routing_batch_size,
        "max_quant_layers": max_quant_layers,
        "auto_round": {
            "enabled": normalized_method == "auto_round",
            "version": str(auto_round_version).lower(),
            "iters": int(auto_round_iters),
            "lr": auto_round_lr,
            "minmax_lr": auto_round_minmax_lr,
        },
        "dynamic": dynamic,
    }
    result = DeepSeekV4FlashQuantRecipeResult(
        source_model_dir=str(source_path),
        prepared_model_dir=str(prepared_path),
        output_dir=str(output_path),
        method=normalized_method,
        artifact_format=normalized_format,
        model_family="deepseek_v4_flash",
        num_layers=target_layers,
        offload_to_disk=bool(offload_to_disk),
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_path}"
        )
    if not source_path.is_dir():
        raise FileNotFoundError(f"DeepSeek-V4 model directory not found: {source_path}")

    source_config = load_json(source_path / "config.json")
    if source_config.get("model_type") != "deepseek_v4":
        raise ValueError("model_dir must contain a DeepSeek-V4 checkpoint")
    is_native = bool(source_config.get("quantization_config")) or (
        str(source_config.get("expert_dtype", "")).lower() == "fp4"
    )
    if is_native:
        if prepared_path.exists():
            _validate_prepared_model(prepared_path, num_layers=target_layers)
        else:
            prepare_deepseek_v4_flash_bf16(
                model_dir=str(source_path),
                output_dir=str(prepared_path),
                num_layers=target_layers,
                device=preparation_device,
            )
    else:
        prepared_path = source_path
        _validate_prepared_model(prepared_path, num_layers=target_layers)

    import torch
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import (
        AutoRoundVersion,
        ExpertsRoutingBypass,
        MoEConfig,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    texts = _load_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=calibration_path,
        calibration_dataset=calibration_dataset,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
    )
    auto_round_enabled = normalized_method == "auto_round"
    quantize_config = QuantizeConfig(
        bits=int(bits),
        dynamic=dynamic,
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=False,
        device=str(device),
        hessian_mse=bool(hessian_mse),
        wait_for_submodule_finalizers=bool(wait_for_submodule_finalizers),
        moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=moe_routing_batch_size)),
        offload_to_disk=bool(offload_to_disk),
        offload_to_disk_path=resolved_offload_path,
        auto_forward_data_parallel=bool(auto_forward_data_parallel),
        auto_round_enabled=auto_round_enabled,
        auto_round_version=(
            AutoRoundVersion(str(auto_round_version).lower())
            if auto_round_enabled
            else None
        ),
        iters=int(auto_round_iters),
        lr=auto_round_lr if auto_round_enabled else None,
        minmax_lr=auto_round_minmax_lr if auto_round_enabled else None,
    )
    model = GPTQModel.load(
        str(prepared_path),
        quantize_config=quantize_config,
        trust_remote_code=bool(trust_remote_code),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(prepared_path),
        trust_remote_code=bool(trust_remote_code),
    )
    tokenized = tokenize_deepseek_v4_flash_calibration(
        texts,
        tokenizer=tokenizer,
        seqlen=seqlen,
    )
    model.quantize(tokenized, batch_size=int(batch_size))
    model.save(str(output_path))
    # Older prepared target checkpoints may have omitted the native draft.
    # Repair from the recipe's original source without rerunning quantization.
    from gptqmodel.utils.deepseek_v4_dspark import preserve_deepseek_v4_dspark

    dspark = preserve_deepseek_v4_dspark(source_path, output_path)
    result.provenance["dspark"] = dspark
    result.provenance["mtp_included"] = dspark is not None
    audit_deepseek_v4_flash_quantized_artifact(output_path)
    _write_quant_provenance(output_path, result)
    return result


def _expected_quantized_module_specs(
    config: Mapping[str, Any],
) -> dict[str, tuple[int, int, int]]:
    """Map every expected module to ``(bits, in_features, out_features)``."""

    num_layers = int(config.get("num_hidden_layers", 0))
    num_experts = int(config.get("n_routed_experts", 0))
    num_shared = int(config.get("n_shared_experts", 0))
    hidden = int(config.get("hidden_size", 0))
    expert_hidden = int(config.get("moe_intermediate_size", 0))
    q_rank = int(config.get("q_lora_rank", 0))
    num_heads = int(config.get("num_attention_heads", 0))
    head_dim = int(config.get("head_dim", 0))
    o_groups = int(config.get("o_groups", 0))
    o_rank = int(config.get("o_lora_rank", 0))
    dimensions = {
        "num_hidden_layers": num_layers,
        "n_routed_experts": num_experts,
        "n_shared_experts": num_shared,
        "hidden_size": hidden,
        "moe_intermediate_size": expert_hidden,
        "q_lora_rank": q_rank,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "o_groups": o_groups,
        "o_lora_rank": o_rank,
    }
    invalid = {name: value for name, value in dimensions.items() if value <= 0}
    if invalid:
        raise ValueError(f"invalid DeepSeek-V4 dimensions: {invalid}")

    specs: dict[str, tuple[int, int, int]] = {}
    for layer in range(num_layers):
        root = f"model.layers.{layer}"
        specs.update(
            {
                f"{root}.self_attn.q_a_proj": (8, hidden, q_rank),
                f"{root}.self_attn.q_b_proj": (8, q_rank, num_heads * head_dim),
                f"{root}.self_attn.kv_proj": (8, hidden, head_dim),
                f"{root}.self_attn.o_b_proj": (8, o_groups * o_rank, hidden),
                f"{root}.mlp.shared_experts.gate_proj": (
                    8,
                    hidden,
                    expert_hidden * num_shared,
                ),
                f"{root}.mlp.shared_experts.up_proj": (
                    8,
                    hidden,
                    expert_hidden * num_shared,
                ),
                f"{root}.mlp.shared_experts.down_proj": (
                    8,
                    expert_hidden * num_shared,
                    hidden,
                ),
            }
        )
        for expert in range(num_experts):
            expert_root = f"{root}.mlp.experts.{expert}"
            specs.update(
                {
                    f"{expert_root}.gate_proj": (4, hidden, expert_hidden),
                    f"{expert_root}.up_proj": (4, hidden, expert_hidden),
                    f"{expert_root}.down_proj": (4, expert_hidden, hidden),
                }
            )
    return specs


def audit_deepseek_v4_flash_quantized_artifact(
    model_dir: str | Path,
) -> DeepSeekV4FlashArtifactAudit:
    """Validate every packed tensor without materializing multi-GB weights.

    This is intentionally stronger than checking ``quantize_config.json``:
    every expected module must physically provide GPTQ ``qweight``, ``qzeros``,
    FP16 ``scales`` and ``g_idx`` tensors with the shape implied by W4/W8 and
    group size 64.  Any extra quantized module is rejected as well.
    """

    root = Path(model_dir).expanduser().resolve()
    config = load_json(root / "config.json")
    if config.get("model_type") != "deepseek_v4":
        raise ValueError(f"quantized artifact is not DeepSeek-V4: {root}")
    quant_config_path = root / "quantize_config.json"
    quant_config = (
        load_json(quant_config_path)
        if quant_config_path.is_file()
        else dict(config.get("quantization_config") or {})
    )
    dynamic = dict(quant_config.get("dynamic") or {})
    expert_override = dynamic.get(_ROUTED_EXPERT_PATTERN)
    if (
        int(quant_config.get("bits", 0)) != 8
        or int(quant_config.get("group_size", 0)) != 64
        or quant_config.get("sym") is not True
        or not isinstance(expert_override, Mapping)
        or int(expert_override.get("bits", 0)) != 4
        or int(expert_override.get("group_size", 0)) != 64
        or expert_override.get("sym") is not True
    ):
        raise ValueError("artifact does not declare symmetric W8/W4 group-size 64")

    index_path = root / "model.safetensors.index.json"
    index = load_json(index_path)
    weight_map = dict(index.get("weight_map") or {})
    if not weight_map:
        raise ValueError(f"empty safetensors weight_map: {index_path}")
    if any(_MTP_KEY.match(name.removeprefix("model.")) for name in weight_map):
        raise ValueError("quantized artifact unexpectedly contains MTP tensors")

    expected_specs = _expected_quantized_module_specs(config)
    actual_prefixes = {
        name[: -len(".qweight")]
        for name in weight_map
        if name.endswith(".qweight")
    }
    missing = sorted(set(expected_specs) - actual_prefixes)
    unexpected = sorted(actual_prefixes - set(expected_specs))
    if missing or unexpected:
        raise ValueError(
            "packed quantized-module inventory mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        if not (root / shard_name).is_file():
            raise FileNotFoundError(f"indexed safetensors shard is missing: {shard_name}")

    expected_suffixes = ("qweight", "qzeros", "scales", "g_idx")
    with ExitStack() as stack:
        readers = {
            shard_name: stack.enter_context(
                safe_open(root / shard_name, framework="pt", device="cpu")
            )
            for shard_name in shard_names
        }
        header_keys = {
            name
            for reader in readers.values()
            for name in reader.keys()
        }
        if header_keys != set(weight_map):
            missing_index = sorted(header_keys - set(weight_map))
            stale_index = sorted(set(weight_map) - header_keys)
            raise ValueError(
                "safetensors index/header mismatch: "
                f"unindexed={missing_index[:10]}, missing={stale_index[:10]}"
            )

        for prefix, (bits, in_features, out_features) in expected_specs.items():
            pack_factor = 32 // bits
            if in_features % pack_factor or out_features % pack_factor:
                raise ValueError(
                    f"{prefix} dimensions are not divisible by W{bits} pack factor"
                )
            groups = (in_features + 63) // 64
            expected = {
                "qweight": ((in_features // pack_factor, out_features), "I32"),
                "qzeros": ((groups, out_features // pack_factor), "I32"),
                "scales": ((groups, out_features), "F16"),
                "g_idx": ((in_features,), "I32"),
            }
            for suffix in expected_suffixes:
                name = f"{prefix}.{suffix}"
                if name not in weight_map:
                    raise ValueError(f"missing packed tensor: {name}")
                tensor_slice = readers[weight_map[name]].get_slice(name)
                actual_shape = tuple(int(value) for value in tensor_slice.get_shape())
                actual_dtype = str(tensor_slice.get_dtype())
                expected_shape, expected_dtype = expected[suffix]
                if actual_shape != expected_shape or actual_dtype != expected_dtype:
                    raise ValueError(
                        f"invalid packed tensor {name}: expected "
                        f"shape={expected_shape}, dtype={expected_dtype}; got "
                        f"shape={actual_shape}, dtype={actual_dtype}"
                    )

    w4_count = sum(bits == 4 for bits, _, _ in expected_specs.values())
    w8_count = sum(bits == 8 for bits, _, _ in expected_specs.values())
    return DeepSeekV4FlashArtifactAudit(
        model_dir=str(root),
        num_layers=int(config["num_hidden_layers"]),
        tensor_count=len(weight_map),
        shard_count=len(shard_names),
        quantized_module_count=len(expected_specs),
        w4_module_count=w4_count,
        w8_module_count=w8_count,
    )


def _build_prepared_config(config: Mapping[str, Any], *, num_layers: int) -> dict:
    prepared = dict(config)
    prepared.pop("quantization_config", None)
    prepared.pop("expert_dtype", None)
    prepared["torch_dtype"] = "bfloat16"
    prepared["num_hidden_layers"] = int(num_layers)
    ratios = list(prepared.get("compress_ratios") or [])
    if ratios:
        if len(ratios) < int(num_layers):
            raise ValueError(
                f"compress_ratios has {len(ratios)} entries, need {num_layers}"
            )
        prepared["compress_ratios"] = ratios[: int(num_layers)]
    prepared["num_nextn_predict_layers"] = 0
    prepared["dspark_target_layer_ids"] = [
        int(layer_index)
        for layer_index in prepared.get("dspark_target_layer_ids", [])
        if int(layer_index) < int(num_layers)
    ]
    if "num_hash_layers" in prepared:
        prepared["num_hash_layers"] = min(
            int(prepared["num_hash_layers"]),
            int(num_layers),
        )
    return prepared


def _validate_prepared_model(path: Path, *, num_layers: int | None) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Prepared config not found: {config_path}")
    config = load_json(config_path)
    if config.get("model_type") != "deepseek_v4":
        raise ValueError(f"Prepared model is not DeepSeek-V4: {path}")
    if config.get("quantization_config"):
        raise ValueError(f"Prepared model is still natively quantized: {path}")
    if str(config.get("expert_dtype", "")).lower() == "fp4":
        raise ValueError(f"Prepared model still declares FP4 experts: {path}")
    if num_layers is not None and int(config.get("num_hidden_layers", -1)) != int(
        num_layers
    ):
        raise ValueError(
            f"Prepared model has {config.get('num_hidden_layers')} layers, expected {num_layers}"
        )


def _validate_quant_recipe_args(
    *,
    method: str,
    artifact_format: str,
    bits: int,
    expert_bits: int,
    group_size: int,
    sym: bool,
    batch_size: int,
    nsamples: int,
    seqlen: int,
    num_layers: int | None,
    max_quant_layers: int | None,
    moe_routing_batch_size: int | None,
    auto_round_version: str,
    auto_round_iters: int,
    auto_round_lr: float | None,
    auto_round_minmax_lr: float | None,
) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            "DeepSeek-V4-Flash recipe method must be 'gptq' or 'auto_round'"
        )
    if artifact_format != "gptqmodel_hf":
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if (int(bits), int(expert_bits), int(group_size)) != (8, 4, 64):
        raise ValueError(
            "DeepSeek-V4-Flash profile requires base W8, routed experts W4, "
            "and group_size=64"
        )
    if sym is not True:
        raise ValueError("DeepSeek-V4-Flash profile requires symmetric quantization")
    if int(batch_size) <= 0 or int(nsamples) <= 0 or int(seqlen) <= 0:
        raise ValueError("batch_size, nsamples, and seqlen must be positive")
    if num_layers is not None and int(num_layers) <= 0:
        raise ValueError("num_layers must be positive when specified")
    if max_quant_layers is not None and int(max_quant_layers) < 0:
        raise ValueError("max_quant_layers must be non-negative")
    if moe_routing_batch_size is not None and int(moe_routing_batch_size) <= 0:
        raise ValueError("moe_routing_batch_size must be positive when specified")
    if str(auto_round_version).lower() not in {"v1", "v2"}:
        raise ValueError("auto_round_version must be 'v1' or 'v2'")
    if int(auto_round_iters) < 0:
        raise ValueError("auto_round_iters must be non-negative")
    if auto_round_lr is not None and float(auto_round_lr) <= 0:
        raise ValueError("auto_round_lr must be positive when specified")
    if auto_round_minmax_lr is not None and float(auto_round_minmax_lr) <= 0:
        raise ValueError("auto_round_minmax_lr must be positive when specified")


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized == "autoround":
        normalized = "auto_round"
    return normalized


def _resolve_offload_path(
    *,
    output_path: Path,
    offload_to_disk: bool,
    offload_path: str | None,
) -> str | None:
    if not offload_to_disk:
        return None
    path = (
        Path(offload_path).expanduser()
        if offload_path
        else output_path.parent / f"{output_path.name}_offload"
    )
    return str(path.resolve())


def _load_calibration_texts(
    *,
    calibration_data: Iterable[str | Mapping[str, Any]] | None,
    calibration_jsonl: Path | None,
    calibration_dataset: str | None,
    calibration_text_key: str,
    nsamples: int,
) -> list[str]:
    if calibration_data is not None:
        source = calibration_data
    elif calibration_jsonl is not None:
        if not calibration_jsonl.is_file():
            raise FileNotFoundError(f"Calibration JSONL not found: {calibration_jsonl}")

        def _jsonl_records():
            with calibration_jsonl.open(encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)

        source = _jsonl_records()
    else:
        if not calibration_dataset:
            raise ValueError(
                "Provide calibration_data, calibration_jsonl, or calibration_dataset"
            )
        from datasets import load_dataset

        source = load_dataset(calibration_dataset, split="train")

    texts: list[str] = []
    for row_index, record in enumerate(source, 1):
        text = record if isinstance(record, str) else record.get(calibration_text_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Invalid calibration field {calibration_text_key!r} at row {row_index}"
            )
        texts.append(text)
        if len(texts) >= int(nsamples):
            break
    if len(texts) < int(nsamples):
        raise RuntimeError(
            f"Calibration source yielded {len(texts)} samples, need {nsamples}"
        )
    return texts


def _write_quant_provenance(
    output_path: Path,
    result: DeepSeekV4FlashQuantRecipeResult,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "deepseek_v4_flash_quant_recipe_provenance.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "DeepSeekV4FlashArtifactAudit",
    "DeepSeekV4FlashPreparationResult",
    "DeepSeekV4FlashQuantRecipeResult",
    "audit_deepseek_v4_flash_quantized_artifact",
    "build_deepseek_v4_flash_dynamic",
    "prepare_deepseek_v4_flash_bf16",
    "quantize_deepseek_v4_flash",
    "select_deepseek_v4_flash_checkpoint_keys",
    "tokenize_deepseek_v4_flash_calibration",
]
