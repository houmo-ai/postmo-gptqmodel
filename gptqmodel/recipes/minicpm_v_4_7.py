"""Stable GPTQ/AutoRound recipes for MiniCPM-V-4.7 text backbones.

MiniCPM-V-4.7 wraps a Qwen3.5 Dense or Qwen3.5 MoE language tower behind a
remote-code multimodal host. GPTQModel cannot load that host directly, so these
recipes own:

* Dense: extract ``Qwen3_5ForCausalLM`` weights → GPTQ → Torch dequant float bin
* MoE: assemble in-memory ``Qwen3_5MoeForCausalLM`` → AutoRound mixed bits →
  restore packed experts → float CausalLM bin

Downstream ModelZoo adapters only resolve repository resources / workflow
kwargs and apply the resulting float CausalLM weights onto the MiniCPM host.
"""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ATTN_LINEAR_SUFFIXES = (
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
_SHARED_EXPERT_SUFFIXES = (
    ".mlp.shared_expert.gate_proj",
    ".mlp.shared_expert.up_proj",
    ".mlp.shared_expert.down_proj",
)
_EXPERT_UP_GATE_SUFFIXES = (".gate_proj", ".up_proj")
_CAUSAL_EXPERT_GATE_UP_PATTERN = "model.layers.*.mlp.experts.*.(gate_proj|up_proj)"
_CAUSAL_EXPERT_DOWN_PATTERN = "model.layers.*.mlp.experts.*.down_proj"
_AUTOROUND_LINEAR_ATTRS = (
    "bits",
    "group_size",
    "sym",
    "data_type",
    "act_bits",
    "act_group_size",
    "act_sym",
    "act_data_type",
    "act_dynamic",
    "super_bits",
    "super_group_size",
    "transform_config",
    "scale",
    "zp",
    "weight_global_scale",
    "scale_dtype",
    "q_scale_thresh",
    "tuning_device",
    "fixed_by_user",
)


@dataclass(frozen=True)
class MiniCPMV47QuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    topology: str
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


def quantize_minicpm_v_4_7_gptq(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    artifact_format: str = "float_causal_bin",
    bits: int = 4,
    group_size: int = 64,
    damp_percent: float = 0.01,
    nsamples: int = 128,
    seqlen: int = 1024,
    sampling_seed: int | None = None,
    calibration_jsonl: str | None = None,
    calibration_data: Iterable[str] | None = None,
    llm_backbone_dir: str | None = None,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    dry_run: bool = False,
) -> MiniCPMV47QuantRecipeResult:
    """Extract Dense Qwen3.5 text, GPTQ-quantize, and emit float CausalLM weights."""

    del device  # GPTQModel selects CUDA devices from the process environment.
    method = str(method).lower().replace("-", "_")
    artifact_format = str(artifact_format).lower()
    if method not in {"gptq", "gptqmodel"}:
        raise ValueError("MiniCPM-V-4.7 Dense recipe method must be 'gptq'")
    if artifact_format != "float_causal_bin":
        raise ValueError("artifact_format must be 'float_causal_bin'")
    if int(group_size) != 64:
        raise ValueError(f"MiniCPM-V-4.7 GPTQ requires group_size=64 for XH2A, got {group_size}")
    if int(bits) <= 0 or int(nsamples) <= 0 or int(seqlen) <= 0:
        raise ValueError("bits/nsamples/seqlen must be positive")

    model_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    precision = {
        "backbone": "qwen3_5",
        "bits": int(bits),
        "group_size": int(group_size),
        "desc_act": False,
        "damp_percent": float(damp_percent),
    }
    calibration = {
        "jsonl": None if calibration_jsonl in (None, "") else str(Path(str(calibration_jsonl)).expanduser()),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "sampling_seed": sampling_seed,
    }
    provenance = {
        "recipe": "gptqmodel.recipes.minicpm_v_4_7.quantize_minicpm_v_4_7_gptq",
        "backend": "direct_gptqmodel_api",
        "dry_run": bool(dry_run),
        "trust_remote_code": bool(trust_remote_code),
        "llm_backbone_dir": llm_backbone_dir,
    }
    dequant_dir = output_path / "gptq_llm_dequant"
    result = MiniCPMV47QuantRecipeResult(
        output_dir=str(dequant_dir),
        method="gptq",
        artifact_format=artifact_format,
        model_family="minicpm_v_4_7",
        topology="dense",
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    samples = _resolve_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=calibration_jsonl,
        nsamples=nsamples,
        sampling_seed=sampling_seed,
    )
    if len(samples) < 16:
        print(
            f"[GPTQ] warning: calibration set has only {len(samples)} sample(s); "
            "more samples generally improve quantization quality"
        )

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    output_path.mkdir(parents=True, exist_ok=True)
    quantize_config = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        desc_act=False,
        damp_percent=float(damp_percent),
        offload_to_disk_path=str(output_path / "gptqmodel_offload"),
    )

    backbone_dir = output_path / "llm_backbone"
    if llm_backbone_dir and Path(llm_backbone_dir).is_dir() and (Path(llm_backbone_dir) / "pytorch_model.bin").is_file():
        shutil.copytree(Path(llm_backbone_dir), backbone_dir, dirs_exist_ok=True)
    else:
        _export_dense_llm_backbone(str(model_path), backbone_dir)

    model = GPTQModel.from_pretrained(
        str(backbone_dir),
        quantize_config=quantize_config,
        trust_remote_code=bool(trust_remote_code),
    )
    model.quantize(samples, batch_size=1, calibration_concat_size=int(seqlen))

    quanted_model_dir = output_path / "gptq_llm"
    model.save_quantized(str(quanted_model_dir))
    _dequantize_gptq_llm(str(quanted_model_dir), quantize_config, dequant_dir)
    _write_provenance(dequant_dir, result)
    return result


def quantize_minicpm_v_4_7_autoround(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "autoround",
    artifact_format: str = "float_causal_bin",
    bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    data_type: str = "int_sym_xh",
    iters: int = 400,
    nsamples: int = 256,
    seqlen: int = 2048,
    batch_size: int = 8,
    gradient_accumulate_steps: int = 1,
    seed: int = 42,
    dataset: str = "NeelNanda/pile-10k",
    device: str = "cuda:0",
    device_map: str = "0",
    low_gpu_mem_usage: bool = True,
    save_quantized: bool = False,
    export_format: str = "auto_round:gptqmodel",
    attn_bits: int | None = 8,
    shared_expert_bits: int | None = 8,
    expert_bits: int | None = None,
    expert_up_gate_bits: int | None = None,
    expert_down_bits: int | None = None,
    dry_run: bool = False,
) -> MiniCPMV47QuantRecipeResult:
    """AutoRound-quantize MiniCPM-V-4.7 MoE text and emit float CausalLM weights."""

    method = str(method).lower().replace("-", "_")
    artifact_format = str(artifact_format).lower()
    if method not in {"autoround", "auto_round"}:
        raise ValueError("MiniCPM-V-4.7 MoE recipe method must be 'autoround'")
    if artifact_format != "float_causal_bin":
        raise ValueError("artifact_format must be 'float_causal_bin'")
    if int(group_size) != 64:
        raise ValueError(f"MiniCPM-V-4.7 AutoRound requires group_size=64 for XH2A, got {group_size}")
    if data_type in {"int_sym_gptq", "int_sym_xh"} and not sym:
        raise ValueError(f"data_type={data_type} requires sym=True")

    model_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    expert_bits = int(bits) if expert_bits is None else int(expert_bits)
    precision = {
        "backbone": "qwen3_5_moe",
        "bits": int(bits),
        "group_size": int(group_size),
        "symmetric": bool(sym),
        "data_type": str(data_type),
        "attn_bits": attn_bits,
        "shared_expert_bits": shared_expert_bits,
        "expert_bits": expert_bits,
        "expert_up_gate_bits": expert_up_gate_bits,
        "expert_down_bits": expert_down_bits,
    }
    calibration = {
        "dataset": str(dataset),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "iters": int(iters),
    }
    provenance = {
        "recipe": "gptqmodel.recipes.minicpm_v_4_7.quantize_minicpm_v_4_7_autoround",
        "backend": "direct_autoround_api",
        "dry_run": bool(dry_run),
        "device": str(device),
        "device_map": str(device_map),
        "batch_size": int(batch_size),
        "gradient_accumulate_steps": int(gradient_accumulate_steps),
        "seed": int(seed),
        "low_gpu_mem_usage": bool(low_gpu_mem_usage),
        "save_quantized": bool(save_quantized),
        "format": str(export_format),
    }
    dequant_dir = output_path / "autoround_llm_dequant"
    result = MiniCPMV47QuantRecipeResult(
        output_dir=str(dequant_dir),
        method="autoround",
        artifact_format=artifact_format,
        model_family="minicpm_v_4_7",
        topology="moe",
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    import torch
    from auto_round import AutoRound
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    output_path.mkdir(parents=True, exist_ok=True)
    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    dequant_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AutoRound] loading MiniCPM-V-4.7 MoE host from {model_path}")
    host = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    causal = _build_moe_causal_from_host(host)
    del host
    gc.collect()

    layer_config = _build_moe_layer_config(
        causal,
        llm_group_size=int(group_size),
        attn_bits=attn_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        expert_up_gate_bits=expert_up_gate_bits,
        expert_down_bits=expert_down_bits,
    )
    _print_moe_override_summary(layer_config)

    ar_device = _normalize_autoround_device(device)
    print(
        f"[AutoRound] quantize CausalLM bits={bits} group_size={group_size} "
        f"data_type={data_type} iters={iters} nsamples={nsamples} dataset={dataset} "
        f"device={ar_device} device_map={device_map}"
    )
    ar = AutoRound(
        model=causal,
        tokenizer=tokenizer,
        bits=int(bits),
        group_size=int(group_size),
        sym=bool(sym),
        data_type=str(data_type),
        iters=int(iters),
        seqlen=int(seqlen),
        nsamples=int(nsamples),
        batch_size=int(batch_size),
        gradient_accumulate_steps=int(gradient_accumulate_steps),
        dataset=str(dataset),
        seed=int(seed),
        layer_config=layer_config or None,
        device=ar_device,
        device_map=device_map,
        low_gpu_mem_usage=bool(low_gpu_mem_usage),
    )
    quantized_model, _ = ar.quantize()
    if quantized_model is not None:
        causal = quantized_model

    if save_quantized:
        quanted_dir = output_path / "autoround_llm"
        quanted_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AutoRound] optional save_quantized -> {quanted_dir} format={export_format}")
        ar.save_quantized(str(quanted_dir), format=str(export_format))

    _strip_autoround_linear_attrs(causal)
    _restore_autoround_moe_blocks(causal)
    _emit_causal_float_bin(causal, dequant_dir)
    _write_provenance(dequant_dir, result)

    del ar, causal, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _resolve_calibration_texts(
    *,
    calibration_data: Iterable[str] | None,
    calibration_jsonl: str | None,
    nsamples: int,
    sampling_seed: int | None,
) -> list[str]:
    if calibration_data is not None:
        texts = [str(item) for item in calibration_data if str(item).strip()]
        return texts[: int(nsamples)]
    if not calibration_jsonl:
        raise ValueError("Provide calibration_data or calibration_jsonl for MiniCPM-V-4.7 GPTQ")
    path = Path(str(calibration_jsonl)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"calibration jsonl not found: {path}")
    records: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if text:
                records.append(text)
    if not records:
        raise ValueError(f"calibration jsonl has no usable 'text' samples: {path}")
    if sampling_seed is None or len(records) <= nsamples:
        return records[: int(nsamples)]
    import numpy as np

    indices = np.random.default_rng(int(sampling_seed)).choice(len(records), size=int(nsamples), replace=False)
    return [records[int(index)] for index in indices]


def _export_dense_llm_backbone(model_dir: str, backbone_dir: Path) -> None:
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    host_cfg = json.loads(Path(model_dir, "config.json").read_text(encoding="utf-8"))
    text_cfg_dict = host_cfg.get("text_config")
    if not isinstance(text_cfg_dict, Mapping):
        raise ValueError(f"MiniCPM-V-4.7 config missing text_config: {model_dir}")

    text_cfg = Qwen3_5TextConfig(**dict(text_cfg_dict))
    text_cfg.architectures = ["Qwen3_5ForCausalLM"]
    text_cfg.save_pretrained(str(backbone_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(str(backbone_dir))
    state = _load_dense_causal_state(model_dir)
    torch.save(state, backbone_dir / "pytorch_model.bin")


def _load_dense_causal_state(model_dir: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    model_path = Path(model_dir)
    single = model_path / "model.safetensors"
    index = model_path / "model.safetensors.index.json"
    if single.is_file():
        shard_map = {key: str(single) for key in _iter_safetensor_keys(single)}
    elif index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"invalid safetensors index: {index}")
        shard_map = {
            str(key): str(model_path / shard)
            for key, shard in weight_map.items()
            if str(key).startswith("model.language_model.") or str(key) == "lm_head.weight"
        }
    else:
        raise FileNotFoundError(f"MiniCPM-V-4.7 weights not found under {model_dir}")

    state: dict[str, Any] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in shard_map.items():
        if not (key.startswith("model.language_model.") or key == "lm_head.weight"):
            continue
        by_shard.setdefault(shard, []).append(key)

    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt") as handle:
            for key in keys:
                tensor = handle.get_tensor(key).detach().cpu().clone()
                if key == "lm_head.weight":
                    state["lm_head.weight"] = tensor
                else:
                    state["model." + key[len("model.language_model.") :]] = tensor

    if "lm_head.weight" not in state:
        raise KeyError(f"lm_head.weight missing from MiniCPM-V-4.7 checkpoint: {model_dir}")
    if not any(key.startswith("model.") for key in state):
        raise KeyError(f"model.language_model.* weights missing from MiniCPM-V-4.7 checkpoint: {model_dir}")
    return state


def _iter_safetensor_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return list(handle.keys())


def _dequantize_gptq_llm(quanted_model_dir: str, quantize_config: Any, dequant_dir: Path) -> None:
    import torch
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import dequantize_model

    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    dequant_dir.mkdir(parents=True, exist_ok=True)

    model = GPTQModel.from_quantized(
        quanted_model_dir,
        quantize_config=quantize_config,
        trust_remote_code=True,
        backend=BACKEND.TORCH,
    )
    model = dequantize_model(model)
    inner = model.model
    state = {key: value.detach().cpu().clone() for key, value in inner.state_dict().items()}
    torch.save(state, dequant_dir / "pytorch_model.bin")
    shutil.copy2(Path(quanted_model_dir) / "config.json", dequant_dir / "config.json")


def _build_moe_causal_from_host(host: Any) -> Any:
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    text_cfg = host.config.text_config
    if isinstance(text_cfg, Qwen3_5MoeTextConfig):
        causal_cfg = text_cfg
    else:
        text_dict = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else dict(text_cfg)
        text_dict.pop("architectures", None)
        causal_cfg = Qwen3_5MoeTextConfig(**text_dict)
    causal_cfg.architectures = ["Qwen3_5MoeForCausalLM"]

    with torch.device("meta"):
        causal = Qwen3_5MoeForCausalLM(causal_cfg)
    causal.model = host.model.language_model
    causal.lm_head = host.lm_head
    if getattr(host, "generation_config", None) is not None:
        causal.generation_config = host.generation_config
    causal.eval()
    return causal


def _build_moe_layer_config(
    model: Any,
    *,
    llm_group_size: int,
    attn_bits: int | None,
    shared_expert_bits: int | None,
    expert_bits: int | None,
    expert_up_gate_bits: int | None,
    expert_down_bits: int | None,
) -> dict[str, dict[str, int]]:
    import torch

    if all(
        value is None
        for value in (attn_bits, shared_expert_bits, expert_bits, expert_up_gate_bits, expert_down_bits)
    ):
        return {}

    layer_config: dict[str, dict[str, int]] = {}
    saw_expert_linear = False
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if attn_bits is not None and name.endswith(_ATTN_LINEAR_SUFFIXES):
            layer_config[name] = {"bits": int(attn_bits), "group_size": int(llm_group_size)}
            continue
        if shared_expert_bits is not None and name.endswith(_SHARED_EXPERT_SUFFIXES):
            layer_config[name] = {"bits": int(shared_expert_bits), "group_size": int(llm_group_size)}
            continue
        if ".mlp.experts." not in name:
            continue
        saw_expert_linear = True
        if expert_down_bits is not None and name.endswith(".down_proj"):
            layer_config[name] = {"bits": int(expert_down_bits), "group_size": int(llm_group_size)}
            continue
        if expert_up_gate_bits is not None and name.endswith(_EXPERT_UP_GATE_SUFFIXES):
            layer_config[name] = {"bits": int(expert_up_gate_bits), "group_size": int(llm_group_size)}
            continue
        if expert_bits is not None and name.endswith((".gate_proj", ".up_proj", ".down_proj")):
            layer_config[name] = {"bits": int(expert_bits), "group_size": int(llm_group_size)}

    if not saw_expert_linear and any(
        value is not None for value in (expert_bits, expert_up_gate_bits, expert_down_bits)
    ):
        gate_up_bits = expert_up_gate_bits if expert_up_gate_bits is not None else expert_bits
        down_bits = expert_down_bits if expert_down_bits is not None else expert_bits
        if gate_up_bits is not None:
            layer_config[_CAUSAL_EXPERT_GATE_UP_PATTERN] = {
                "bits": int(gate_up_bits),
                "group_size": int(llm_group_size),
            }
        if down_bits is not None:
            layer_config[_CAUSAL_EXPERT_DOWN_PATTERN] = {
                "bits": int(down_bits),
                "group_size": int(llm_group_size),
            }
    return layer_config


def _print_moe_override_summary(layer_config: Mapping[str, Any]) -> None:
    if not layer_config:
        print("[AutoRound] LLM mixed-bit overrides: none")
        return
    counts = {"attention": 0, "shared_expert": 0, "experts": 0, "patterns": 0}
    for name in layer_config:
        if any(token in name for token in ("*", "|", "(", ")")):
            counts["patterns"] += 1
        if name.endswith(_ATTN_LINEAR_SUFFIXES):
            counts["attention"] += 1
        elif any(token in name for token in _SHARED_EXPERT_SUFFIXES):
            counts["shared_expert"] += 1
        elif ".mlp.experts." in name:
            counts["experts"] += 1
    print(
        "[AutoRound] LLM mixed-bit overrides: "
        f"attention={counts['attention']}, shared_expert={counts['shared_expert']}, "
        f"experts={counts['experts']}, patterns={counts['patterns']}"
    )


def _strip_autoround_linear_attrs(model: Any) -> None:
    import torch

    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for attr in _AUTOROUND_LINEAR_ATTRS:
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except Exception:
                    pass
        for attr_name in list(vars(module)):
            if attr_name.startswith("w_") or attr_name.startswith("act_"):
                try:
                    delattr(module, attr_name)
                except Exception:
                    pass


def _restore_autoround_qwen3_5_moe_sparse_block(module: Any) -> Any:
    import torch
    import torch.nn as nn
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeExperts,
        Qwen3_5MoeSparseMoeBlock,
    )

    restored = object.__new__(Qwen3_5MoeSparseMoeBlock)
    nn.Module.__init__(restored)
    restored.gate = module.gate
    if isinstance(module.experts, nn.ModuleList):
        if not module.experts:
            raise ValueError("LinearQwen3_5MoeSparseMoeBlock must contain at least one expert.")
        first_expert = module.experts[0]
        gate_rows = first_expert.gate_proj.weight.shape[0]
        up_rows = first_expert.up_proj.weight.shape[0]
        gate_up_proj = first_expert.gate_proj.weight.new_empty(
            (len(module.experts), gate_rows + up_rows, *first_expert.gate_proj.weight.shape[1:])
        )
        down_proj = first_expert.down_proj.weight.new_empty((len(module.experts), *first_expert.down_proj.weight.shape))
        with torch.no_grad():
            for expert_idx, expert in enumerate(module.experts):
                gate_up_proj[expert_idx, :gate_rows].copy_(expert.gate_proj.weight)
                gate_up_proj[expert_idx, gate_rows:].copy_(expert.up_proj.weight)
                down_proj[expert_idx].copy_(expert.down_proj.weight)
        restored_experts = object.__new__(Qwen3_5MoeExperts)
        nn.Module.__init__(restored_experts)
        restored_experts.num_experts = len(module.experts)
        restored_experts.hidden_dim = first_expert.gate_proj.weight.shape[1]
        restored_experts.intermediate_dim = gate_rows
        restored_experts.config = first_expert.config
        restored_experts.act_fn = first_expert.act_fn
        restored_experts.gate_up_proj = nn.Parameter(
            gate_up_proj,
            requires_grad=any(
                expert.gate_proj.weight.requires_grad or expert.up_proj.weight.requires_grad
                for expert in module.experts
            ),
        )
        restored_experts.down_proj = nn.Parameter(
            down_proj,
            requires_grad=any(expert.down_proj.weight.requires_grad for expert in module.experts),
        )
        restored.experts = restored_experts
    else:
        restored.experts = module.experts
    restored.shared_expert = module.shared_expert
    restored.shared_expert_gate = module.shared_expert_gate
    restored.train(module.training)
    return restored


def _restore_autoround_moe_blocks(model: Any) -> None:
    block_names = [
        name
        for name, module in model.named_modules()
        if type(module).__name__ == "LinearQwen3_5MoeSparseMoeBlock"
    ]
    print(f"[AutoRound] restoring {len(block_names)} LinearQwen3_5MoeSparseMoeBlock modules")
    for name in block_names:
        module = model.get_submodule(name) if name else model
        restored = _restore_autoround_qwen3_5_moe_sparse_block(module)
        if not name:
            raise RuntimeError("unexpected root-level LinearQwen3_5MoeSparseMoeBlock")
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, restored)


def _emit_causal_float_bin(model: Any, dequant_dir: Path) -> None:
    import torch

    dequant_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, dequant_dir / "pytorch_model.bin")
    if hasattr(model, "config") and hasattr(model.config, "to_json_file"):
        model.config.to_json_file(str(dequant_dir / "config.json"))
    print(f"[AutoRound] wrote float CausalLM dequant to {dequant_dir / 'pytorch_model.bin'}")


def _normalize_autoround_device(device: str) -> str:
    token = str(device).strip()
    if token.isdigit():
        return f"cuda:{token}"
    return token


def _write_provenance(output_path: Path, result: MiniCPMV47QuantRecipeResult) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": result.method,
        "artifact_format": result.artifact_format,
        "model_family": result.model_family,
        "topology": result.topology,
        "precision": result.precision,
        "calibration": result.calibration,
        "provenance": result.provenance,
    }
    (output_path / "minicpm_v_4_7_quant_recipe_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "MiniCPMV47QuantRecipeResult",
    "quantize_minicpm_v_4_7_autoround",
    "quantize_minicpm_v_4_7_gptq",
]
