# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MoE MTP calibration, independent of deployment runtimes.

Only routed experts are packed. The main model (including its head) and all
other MTP tensors remain unchanged. MTP has its own versioned GPTQ V2 payload
contract because Transformers intentionally does not instantiate this module.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from .lm_head import _checkpoint_weight_map, read_checkpoint_config


@dataclass(frozen=True)
class MTPQuantizationConfig:
    algorithm: str = "inherit"
    expert_bits: int = 4
    group_size: int = 64
    data_type: str = "int_sym_xh"
    iters: int = 400
    batch_tokens: int = 512
    max_tokens_per_expert: int = 8192
    validation_nsamples: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.algorithm not in ("inherit", "autoround", "gptq"):
            raise ValueError("MTP algorithm must be inherit, autoround or gptq")
        if (self.expert_bits, self.group_size, self.data_type) != (4, 64, "int_sym_xh"):
            raise ValueError("MTP supports routed experts W4/G64/int_sym_xh only; other weights use export W8")
        for name in ("iters", "batch_tokens", "max_tokens_per_expert", "validation_nsamples"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"MTP {name} must be a positive integer")


def resolve_mtp_algorithm(source, requested="inherit", *, backbone_algorithm=None):
    """Follow the actual backbone producer, including AutoRound's GPTQ exports."""
    from .lm_head import resolve_head_algorithm

    if requested not in ("inherit", "autoround", "gptq"):
        raise ValueError("MTP algorithm must be inherit, autoround or gptq")
    if backbone_algorithm is not None:
        actual = backbone_algorithm
    else:
        config = read_checkpoint_config(source)
        if config.get("quantization_config"):
            try:
                actual = resolve_head_algorithm(source)
            except ValueError as exc:
                raise ValueError("Cannot identify the backbone producer for MTP calibration") from exc
        elif config.get("mtp_quantization"):
            actual = config["mtp_quantization"].get("algorithm")
        elif requested in ("autoround", "gptq"):
            actual = requested
        else:
            raise ValueError("Native HF MTP requires an explicit algorithm or the full backbone recipe")
    if actual not in ("autoround", "gptq"):
        raise ValueError(f"Unsupported MTP backbone algorithm: {actual}")
    if requested != "inherit" and requested != actual:
        raise ValueError(f"MTP algorithm {requested} conflicts with backbone producer {actual}")
    return actual


class CheckpointTensors:
    def __init__(self, source):
        self.root = Path(source)
        self.mapping, self.index = _checkpoint_weight_map(self.root)
        self.config = read_checkpoint_config(self.root)

    def get(self, key):
        with safe_open(self.root / self.mapping[key], framework="pt", device="cpu") as tensors:
            return tensors.get_tensor(key)


def mtp_text_config(config):
    text = config.get("text_config", config)
    if text.get("model_type") != "qwen3_5_moe_text" or text.get("mtp_num_hidden_layers", 1) != 1:
        raise ValueError("MTP calibration requires Qwen3.5/3.6 MoE with exactly one MTP layer")
    if (config.get("quantization_config") or {}).get("rotation"):
        raise ValueError("MTP calibration does not support rotated checkpoints")
    return text


def load_packed_linear(reader, prefix, *, bits, group_size, in_features, out_features, qzero_format=2):
    """Read and validate canonical payloads without loading the main model."""
    from ..nn_modules.qlinear.torch import TorchQuantLinear
    from ..utils.model import convert_gptq_v1_to_v2_format_module

    module = TorchQuantLinear(bits=bits, group_size=group_size, sym=True, desc_act=False,
                             in_features=in_features, out_features=out_features)
    payload = {name: reader.get(prefix + "." + name) for name in ("qweight", "qzeros", "scales", "g_idx")}
    expected = module.state_dict()
    for name, value in payload.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Invalid MTP packed tensor {prefix}.{name}: {value.shape}/{value.dtype}")
    if not torch.equal(payload["g_idx"], torch.arange(in_features, dtype=torch.int32) // group_size):
        raise ValueError(f"MTP requires sequential G64 groups: {prefix}")
    if not torch.isfinite(payload["scales"]).all() or not (payload["scales"] > 0).all():
        raise ValueError(f"MTP scales must be finite and positive: {prefix}")
    module.load_state_dict(payload, strict=True)
    module.qzero_format(format=qzero_format)
    if qzero_format == 1:
        convert_gptq_v1_to_v2_format_module(module, bits=module.bits, pack_dtype=module.pack_dtype)
    shifts = torch.arange(0, 32, module.bits, dtype=torch.int64)
    zeros = (module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << module.bits) - 1)
    if not (zeros == (1 << (bits - 1))).all():
        raise ValueError(f"Packed symmetric zero points are inconsistent: {prefix}")
    if bits != module.bits:
        codes = (module.qweight.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << module.bits) - 1)
        if (codes > (1 << bits) - 1).any():
            raise ValueError(f"Packed codes exceed logical W{bits}: {prefix}")
    return module


def inspect_mtp_checkpoint(source, options):
    reader = CheckpointTensors(source)
    text = mtp_text_config(reader.config)
    metadata = reader.config.get("mtp_quantization")
    prefix = "mtp.layers.0.mlp.experts."
    packed = any(k.startswith(prefix) and k.endswith(".qweight") for k in reader.mapping)
    if not metadata:
        if packed:
            raise ValueError("Packed MTP experts have no mtp_quantization metadata")
        # Loading the native MTP below checks every floating tensor and its shape.
        if not any(k.startswith("mtp.") for k in reader.mapping):
            raise ValueError("Checkpoint has no MTP weights")
        return None
    algorithm = resolve_mtp_algorithm(source, options.algorithm)
    expected = {"version": 1, "algorithm": algorithm, "expert_bits": options.expert_bits,
                "group_size": options.group_size, "data_type": options.data_type,
                "format": "gptq_v2", "scope": "routed_experts", "other_weights": "float"}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Already packed MTP does not match the requested quantization contract")
    expected_keys = set()
    for expert in range(text["num_experts"]):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}{expert}.{proj}"
            ins, outs = text["hidden_size"], text["moe_intermediate_size"]
            if proj == "down_proj":
                ins, outs = outs, ins
            load_packed_linear(reader, key, bits=4, group_size=64, in_features=ins, out_features=outs)
            expected_keys.update(key + "." + suffix for suffix in ("qweight", "qzeros", "scales", "g_idx"))
    actual_keys = {k for k in reader.mapping if k.startswith(prefix)}
    if actual_keys != expected_keys:
        raise ValueError("MTP expert keys are incomplete or contain conflicting floating weights")
    return metadata


class Qwen35MTP(nn.Module):
    """Native HF full-attention draft trunk; output is post-final-norm hidden.

    Input pair at t is main post-norm hidden[t] and embedding(token[t+1]).
    The output is scored with the *same* main head to predict token[t+2].
    """
    def __init__(self, config):
        super().__init__()
        from transformers import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm, Qwen3_5MoeTextRotaryEmbedding,
        )
        values = copy.deepcopy(mtp_text_config(config))
        values.update(num_hidden_layers=1, layer_types=["full_attention"])
        cfg = Qwen3_5MoeTextConfig(**values)
        cfg._attn_implementation = "eager"
        self.config = cfg
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, 0)])
        self.norm = Qwen3_5MoeRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(cfg)

    def forward(self, next_token_embedding, post_norm_hidden, position_ids=None):
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(next_token_embedding),
                                   self.pre_fc_norm_hidden(post_norm_hidden)), dim=-1))
        size = hidden.shape[1]
        if position_ids is None:
            position_ids = torch.arange(size, device=hidden.device).unsqueeze(0)
        mask = torch.full((size, size), torch.finfo(hidden.dtype).min,
                          dtype=hidden.dtype, device=hidden.device).triu(1)[None, None]
        hidden = self.layers[0](hidden, position_embeddings=self.rotary_emb(hidden, position_ids),
                                attention_mask=mask)
        return self.norm(hidden)

    @classmethod
    def from_pretrained(cls, source, dtype=torch.float16):
        reader = CheckpointTensors(source)
        with torch.device("meta"):
            model = cls(reader.config)
        state = {k[4:]: reader.get(k).to(dtype) for k in reader.mapping
                 if k.startswith("mtp.") and ".mlp.experts." not in k}
        text = mtp_text_config(reader.config)
        prefix = "mtp.layers.0.mlp.experts."
        if reader.config.get("mtp_quantization"):
            inspect_mtp_checkpoint(source, MTPQuantizationConfig())
            gates, ups, downs = [], [], []
            for expert in range(text["num_experts"]):
                for proj, output in (("gate_proj", gates), ("up_proj", ups), ("down_proj", downs)):
                    ins, outs = text["hidden_size"], text["moe_intermediate_size"]
                    if proj == "down_proj":
                        ins, outs = outs, ins
                    module = load_packed_linear(reader, f"{prefix}{expert}.{proj}", bits=4, group_size=64,
                                                in_features=ins, out_features=outs)
                    output.append(module.dequantize_weight().T.to(dtype))
            state["layers.0.mlp.experts.gate_up_proj"] = torch.cat((torch.stack(gates), torch.stack(ups)), dim=1)
            state["layers.0.mlp.experts.down_proj"] = torch.stack(downs)
        elif prefix + "gate_up_proj" in reader.mapping:
            for proj in ("gate_up_proj", "down_proj"):
                state["layers.0.mlp.experts." + proj] = reader.get(prefix + proj).to(dtype)
        else:
            pieces = {p: torch.stack([reader.get(f"{prefix}{i}.{p}.weight").to(dtype)
                                      for i in range(text["num_experts"])])
                      for p in ("gate_proj", "up_proj", "down_proj")}
            state["layers.0.mlp.experts.gate_up_proj"] = torch.cat((pieces["gate_proj"], pieces["up_proj"]), dim=1)
            state["layers.0.mlp.experts.down_proj"] = pieces["down_proj"]
        model.load_state_dict(state, strict=True, assign=True)
        # Rotary buffers are non-persistent and must be materialized separately.
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(model.config)
        model.requires_grad_(False).eval()
        return model


def tune_expert(weights, inputs, options, *, device):
    """Joint AutoRound reconstruction of gate/up/down, including their SiLU product."""
    from ..looper.lm_head_auto_round_processor import _autoround_components
    from ..nn_modules.qlinear.torch import TorchQuantLinear
    from .auto_round import build_round_g_idx

    WrapperLinear, SignSGD = _autoround_components()
    with torch.inference_mode(False), torch.enable_grad():
        wrappers, layers = {}, {}
        for name, weight in weights.items():
            weight = weight.detach().to(device=device, dtype=torch.float16).clone()
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
            layer.weight = nn.Parameter(weight, requires_grad=False)
            for key, value in dict(bits=4, group_size=64, sym=True, data_type="int_sym_xh",
                                   act_bits=16, scale_dtype=torch.float16).items():
                setattr(layer, key, value)
            layers[name] = layer
            wrappers[name] = WrapperLinear(layer, device=device, enable_torch_compile=False,
                                           enable_minmax_tuning=True)
        lr = 1.0 / options.iters
        params = [{"params": [value], "lr": lr}
                  for wrapper in wrappers.values() for value in wrapper.params.values()]
        optimizer = SignSGD(params, lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0,
                                                      total_iters=options.iters)
        generator = torch.Generator().manual_seed(options.seed)
        monitor_ids = torch.linspace(0, len(inputs) - 1, min(256, len(inputs))).long()
        monitor = inputs[monitor_ids].to(device=device, dtype=torch.float16).clone()

        def forward(modules, x):
            return modules["down_proj"](F.silu(modules["gate_proj"](x)) * modules["up_proj"](x))

        def snapshot():
            return {name: {k: v.detach().clone() for k, v in wrapper.params.items()}
                    for name, wrapper in wrappers.items()}

        with torch.no_grad():
            reference = forward(layers, monitor).float()
            initial = float(F.mse_loss(forward(wrappers, monitor).float(), reference))
        best, best_step, best_params = initial, 0, snapshot()
        if not math.isfinite(best):
            raise ValueError("MTP AutoRound initial loss is not finite")
        for step in range(options.iters):
            ids = torch.randint(len(inputs), (min(options.batch_tokens, len(inputs)),), generator=generator)
            x = inputs[ids].to(device=device, dtype=torch.float16).clone()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                target = forward(layers, x).float()
            loss = F.mse_loss(forward(wrappers, x).float(), target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite MTP AutoRound loss at iteration {step}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 10 == 0 or step + 1 == options.iters:
                with torch.no_grad():
                    current = float(F.mse_loss(forward(wrappers, monitor).float(), reference))
                if math.isfinite(current) and current < best:
                    best, best_step, best_params = current, step + 1, snapshot()
        packed = {}
        for name, wrapper in wrappers.items():
            with torch.no_grad():
                layer = wrapper.unwrapper(best_params[name])
            rows, cols = layer.weight.shape
            scale = layer.scale.reshape(rows, -1).float().cpu()
            zero = layer.zp
            zero = (zero.reshape(rows, -1).float().cpu() if torch.is_tensor(zero)
                    else torch.full_like(scale, float(zero)))
            module = TorchQuantLinear(bits=4, group_size=64, sym=True, desc_act=False,
                                     in_features=cols, out_features=rows)
            module.pack(layer, scale, zero, build_round_g_idx(cols, 64), workers=1)
            module.qzero_format(format=2)
            packed[name] = {k: v.cpu().contiguous() for k, v in module.state_dict().items()}
        return packed, {"initial_loss": initial, "loss": best, "best_step": best_step, "tokens": len(inputs)}


def calibrate_mtp_experts(model, expert_inputs, options, *, device):
    """Use actual MTP routing; explicitly record bypass data for unvisited experts."""
    inputs = torch.cat(expert_inputs).reshape(-1, model.config.hidden_size).cpu()
    if not len(inputs) or not torch.isfinite(inputs).all():
        raise ValueError("MTP calibration has no finite, non-padding input tokens")
    gate = model.layers[0].mlp.gate.to(device)
    selected = []
    with torch.no_grad():
        for chunk in inputs.split(2048):
            selected.append(gate(chunk.to(device=device, dtype=gate.weight.dtype))[2].cpu())
    selected = torch.cat(selected)
    gate.cpu()
    experts = model.layers[0].mlp.experts
    payload, metrics = {}, []
    generator = torch.Generator().manual_seed(options.seed)
    for expert in range(model.config.num_experts):
        ids = (selected == expert).any(dim=-1).nonzero().flatten()
        routed_tokens = len(ids)
        bypass = routed_tokens == 0
        if bypass:
            warnings.warn(f"MTP expert {expert} received no routed tokens; {options.algorithm} uses sampled MTP inputs in bypass mode")
            ids = torch.arange(len(inputs))
        if len(ids) > options.max_tokens_per_expert:
            ids = ids[torch.randperm(len(ids), generator=generator)[:options.max_tokens_per_expert]]
        gate_weight, up_weight = experts.gate_up_proj[expert].chunk(2, dim=0)
        solver = {"autoround": tune_expert, "gptq": tune_expert_gptq}.get(options.algorithm)
        if solver is None:
            raise ValueError("Resolve the MTP backbone algorithm before calibration")
        packed, stat = solver({"gate_proj": gate_weight, "up_proj": up_weight,
                                    "down_proj": experts.down_proj[expert]}, inputs[ids], options, device=device)
        for proj, tensors in packed.items():
            payload.update({f"mtp.layers.0.mlp.experts.{expert}.{proj}.{key}": value
                            for key, value in tensors.items()})
        stat.update(expert=expert, routed_tokens=routed_tokens, bypass=bypass)
        metrics.append(stat)
        print(f"[mtp] expert {expert + 1}/{model.config.num_experts}: "
              f"{stat['initial_loss']:.7g} -> {stat['loss']:.7g}; routed={routed_tokens}", flush=True)
    return payload, metrics


@torch.no_grad()
def tune_expert_gptq(weights, inputs, options, *, device):
    """GPTQ gate/up first; collect down Hessian from their quantized outputs."""
    from .config import QuantizeConfig, FORMAT
    from .gptq import GPTQ
    from ..nn_modules.qlinear.torch import TorchQuantLinear

    layers, rtn_weights, payload, projections = {}, {}, {}, {}
    monitor = inputs[torch.linspace(0, len(inputs) - 1, min(256, len(inputs))).long()].to(
        device=device, dtype=torch.float16)

    def forward(values, x):
        return F.linear(F.silu(F.linear(x, values["gate_proj"])) * F.linear(x, values["up_proj"]),
                        values["down_proj"])

    floating = {k: v.to(device=device, dtype=torch.float16) for k, v in weights.items()}
    reference = forward(floating, monitor).float()
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = floating[name]
        rows, cols = weight.shape
        layer = nn.Linear(cols, rows, bias=False, device="meta")
        layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
        qcfg = QuantizeConfig(bits=4, group_size=64, sym=True, desc_act=False, act_group_aware=False,
                              data_type=options.data_type, format=FORMAT.GPTQ_V2, failsafe=None,
                              offload_to_disk=False)
        solver = GPTQ(layer, qcfg=qcfg)
        # Bare nn.Linear uses GPTQ's HF compatibility name, whose configure()
        # defaults would otherwise override qcfg.sym with False.
        solver.quantizer.configure(perchannel=True, bits=4, sym=True)
        rtn = torch.empty_like(weight)
        for start in range(0, cols, 64):
            block = weight[:, start:start + 64]
            solver.quantizer.find_params(block, weight=True)
            rtn[:, start:start + 64] = solver.quantizer.quantize(block)
        rtn_weights[name] = rtn
        for chunk in inputs.split(options.batch_tokens):
            x = chunk.to(device=device, dtype=torch.float16)
            if name == "down_proj":
                x = F.silu(layers["gate_proj"](x)) * layers["up_proj"](x)
            solver.add_batch(x.unsqueeze(0), None)
        quant, scales, zeros, g_idx, _, loss, damp, nsamples = solver.quantize()
        layer.weight.copy_(quant)
        module = TorchQuantLinear(bits=4, group_size=64, sym=True, desc_act=False,
                                 in_features=cols, out_features=rows)
        module.pack(layer, scales.cpu(), zeros.cpu(), g_idx.cpu(), workers=1)
        module.qzero_format(format=2)
        payload[name] = {k: v.cpu().contiguous() for k, v in module.state_dict().items()}
        # Replay the actual serialized weights when collecting the down inputs.
        layer.weight.copy_(module.dequantize_weight().T.to(device))
        layers[name] = layer
        projections[name] = {"hessian_tokens": nsamples, "hessian_loss": float(loss), "damp_percent": damp}
        solver.free()
    initial = float(F.mse_loss(forward(rtn_weights, monitor).float(), reference))
    loss = float(F.mse_loss(forward({k: v.weight for k, v in layers.items()}, monitor).float(), reference))
    if not math.isfinite(initial) or not math.isfinite(loss):
        raise ValueError("MTP GPTQ expert reconstruction is not finite")
    return payload, {"initial_loss": initial, "loss": loss, "tokens": len(inputs),
                     "algorithm": "gptq", "true_sequential": True, "projections": projections}


def validate_mtp_payload(model, payload, records, source, *, device):
    """Held-out teacher-forced draft comparison; not live speculative acceptance."""
    from ..nn_modules.qlinear.torch import TorchQuantLinear
    from .config import QuantizeConfig, FORMAT

    reader = CheckpointTensors(source)
    text = reader.config.get("text_config", reader.config)
    if "lm_head.qweight" in reader.mapping:
        config = QuantizeConfig.from_quant_config(reader.config["quantization_config"])
        if config.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
            raise ValueError("MTP validation requires a GPTQ/GPTQ V2 main head")
        values = {k: config.dynamic_get("lm_head", k, getattr(config, k)) for k in ("bits", "group_size")}
        packed = load_packed_linear(reader, "lm_head", **values, in_features=text["hidden_size"],
                                    out_features=text["vocab_size"], qzero_format=1 if config.format == FORMAT.GPTQ else 2)
        head_weight = packed.to(device).dequantize_weight().T.contiguous()
        del packed
    else:
        key = "lm_head.weight"
        if key not in reader.mapping and (reader.config.get("tie_word_embeddings") or text.get("tie_word_embeddings")):
            key = next(k for k in reader.mapping if k.endswith("embed_tokens.weight"))
        head_weight = reader.get(key).to(device=device, dtype=torch.float16)
    expert_weights = model.layers[0].mlp.experts
    for expert in range(model.config.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"mtp.layers.0.mlp.experts.{expert}.{proj}."
            ins, outs = text["hidden_size"], text["moe_intermediate_size"]
            if proj == "down_proj":
                ins, outs = outs, ins
            module = TorchQuantLinear(bits=4, group_size=64, sym=True, desc_act=False,
                                     in_features=ins, out_features=outs)
            module.load_state_dict({k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)})
            module.qzero_format(format=2)
            weight = module.dequantize_weight().T
            with torch.no_grad():
                if proj == "down_proj":
                    expert_weights.down_proj[expert].copy_(weight)
                else:
                    halves = expert_weights.gate_up_proj[expert].chunk(2, dim=0)
                    halves[0 if proj == "gate_proj" else 1].copy_(weight)
    model.to(device).eval()
    tokens, kl_sum, same, fp_match, quant_match = 0, 0.0, 0, 0, 0
    with torch.inference_mode():
        for record in records:
            output = model(record["embedding"].to(device), record["hidden"].to(device)).reshape(-1, text["hidden_size"])
            reference = record["reference"].reshape_as(output).to(device)
            main = record["main_next_hidden"].reshape_as(output).to(device=device, dtype=head_weight.dtype)
            for start in range(0, len(output), 32):
                sl = slice(start, start + 32)
                ref = F.linear(reference[sl], head_weight).float().log_softmax(-1)
                quant = F.linear(output[sl], head_weight).float().log_softmax(-1)
                main_ids = F.linear(main[sl], head_weight).argmax(-1)
                kl_sum += float((ref.exp() * (ref - quant)).sum())
                ref_ids, quant_ids = ref.argmax(-1), quant.argmax(-1)
                same += int((ref_ids == quant_ids).sum())
                fp_match += int((ref_ids == main_ids).sum())
                quant_match += int((quant_ids == main_ids).sum())
                tokens += len(ref)
    model.cpu()
    result = {"scope": "held_out_teacher_forcing", "samples": len(records), "tokens": tokens,
              "draft_fp_to_w4_kl": kl_sum / tokens, "draft_top1_agreement": same / tokens,
              "fp_draft_main_top1_match": fp_match / tokens, "w4_draft_main_top1_match": quant_match / tokens,
              "note": "Floating non-expert MTP weights; not SEFP W8 runtime or live speculative acceptance."}
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError("MTP validation produced non-finite metrics")
    return result


def save_mtp_checkpoint(source, target, payload, options, provenance):
    """Publish atomically; rewrite only shards containing replaced MTP experts."""
    reader = CheckpointTensors(source)
    source, target = reader.root.resolve(), Path(target).resolve()
    if source == target or source in target.parents:
        raise ValueError("MTP output must be outside the source checkpoint")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"MTP output already exists: {target}")
    old = {k for k in reader.mapping if k.startswith("mtp.layers.0.mlp.experts.")}
    affected = {reader.mapping[k] for k in old}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=target.name + ".tmp-", dir=target.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        for path in source.iterdir():
            if path.name in affected:
                continue
            if path.is_dir():
                shutil.copytree(path, stage / path.name)
            else:
                shutil.copy2(path, stage / path.name)
        total = 0
        for filename in set(reader.mapping.values()):
            with safe_open(source / filename, framework="pt", device="cpu") as tensors:
                retained = {k: tensors.get_tensor(k) for k in tensors.keys() if k not in old}
                total += sum(t.numel() * t.element_size() for t in retained.values())
                if filename in affected and retained:
                    save_file(retained, stage / filename, metadata=tensors.metadata())
        mapping = {k: v for k, v in reader.mapping.items() if k not in old}
        if (stage / "model.safetensors").exists():
            (stage / "model.safetensors").rename(stage / "model-backbone.safetensors")
            mapping = {k: "model-backbone.safetensors" if v == "model.safetensors" else v for k, v in mapping.items()}
        filename = "model-mtp-experts.safetensors"
        if (stage / filename).exists():
            raise ValueError(f"Source already contains reserved MTP shard {filename}")
        save_file(payload, stage / filename, metadata={"format": "pt"})
        mapping.update(dict.fromkeys(payload, filename))
        index = copy.deepcopy(reader.index)
        index["weight_map"] = mapping
        index.setdefault("metadata", {})["total_size"] = total + sum(t.numel() * t.element_size() for t in payload.values())
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        # Preserve the original embedded/sidecar layout, just as head completion
        # does. read_checkpoint_config merges sidecars for reading only.
        config = json.loads((source / "config.json").read_text())
        config["mtp_quantization"] = {**asdict(options), "version": 1, "format": "gptq_v2",
                                      "scope": "routed_experts", "other_weights": "float"}
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "qwen35_mtp_recipe_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        inspect_mtp_checkpoint(stage, options)
        if target.exists():
            target.rmdir()
        stage.rename(target)
