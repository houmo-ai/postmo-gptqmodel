"""Sequential high-order calibration of a native dense Qwen MTP decoder block."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .qwen35_mtp import MTPQuantizationConfig, load_packed_linear, resolve_mtp_algorithm


def dense_linears(model):
    return {name: module for name, module in model.named_modules()
            if name.startswith("layers.") and isinstance(module, nn.Linear)}


def inspect_dense_checkpoint(reader, options):
    from .qwen35_mtp import Qwen35MTP
    if options.block_bits is None:
        raise ValueError("Dense MTP requires block_bits (4/5/8), not expert_bits")
    with torch.device("meta"):
        model = Qwen35MTP(reader.config)
    modules = dense_linears(model)
    metadata = reader.config.get("mtp_quantization")
    if metadata is None:
        if any(k.startswith("mtp.") and k.endswith(".qweight") for k in reader.mapping):
            raise ValueError("Packed dense MTP has no metadata")
        if not any(k.startswith("mtp.layers.") for k in reader.mapping):
            raise ValueError("Checkpoint has no MTP weights")
        return None
    expected = {"version": 1, "scope": "dense_block", "block_bits": options.block_bits,
                "fusion_bits": 8, "group_size": 64, "format": "gptq_v2",
                "data_type": "int_sym_xh", "other_weights": "float",
                "algorithm": resolve_mtp_algorithm(reader.root, options.algorithm)}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Already packed dense MTP does not match requested contract")
    expected_keys = set()
    for name, layer in modules.items():
        prefix = "mtp." + name
        load_packed_linear(reader, prefix, bits=options.block_bits, group_size=64,
                          in_features=layer.in_features, out_features=layer.out_features)
        if prefix + ".weight" in reader.mapping:
            raise ValueError("Conflicting packed and floating dense MTP weights")
        expected_keys.update(prefix + "." + suffix for suffix in ("qweight", "qzeros", "scales", "g_idx"))
    actual = {k for k in reader.mapping if k.startswith("mtp.") and k.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))}
    if actual != expected_keys:
        raise ValueError("Incomplete or unexpected dense MTP payload")
    return metadata


def load_dense_state(model, reader, dtype):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    metadata = reader.config.get("mtp_quantization")
    modules = dense_linears(model)
    packed_prefixes = {"mtp." + name for name in modules} if metadata else set()
    state = {key[4:]: reader.get(key).to(dtype) for key in reader.mapping
             if key.startswith("mtp.") and key.rsplit(".", 1)[0] not in packed_prefixes}
    if metadata:
        options = MTPQuantizationConfig(block_bits=metadata["block_bits"], algorithm=metadata["algorithm"])
        inspect_dense_checkpoint(reader, options)
        for name, layer in modules.items():
            packed = load_packed_linear(reader, "mtp." + name, bits=options.block_bits, group_size=64,
                                       in_features=layer.in_features, out_features=layer.out_features)
            state[name + ".weight"] = packed.dequantize_weight().T.to(dtype).contiguous()
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb = Qwen3_5TextRotaryEmbedding(model.config)
    model.requires_grad_(False).eval()


def calibrate_dense(model, records, options, *, device):
    """Collect each Linear's Hessian/reconstruction inputs after preceding updates.

    Norms, shared head/embedding and fusion FC are excluded. Native attention,
    gated MLP and residual paths are replayed for every projection.
    """
    if options.block_bits is None or not records:
        raise ValueError("Dense MTP requires block_bits and non-empty paired calibration inputs")
    payload, metrics = {}, {}
    model.to(device).eval()
    for name, layer in dense_linears(model).items():
        captured, count = [], 0

        def capture(module, args):
            nonlocal count
            remaining = options.max_tokens_per_expert - count
            if remaining > 0:
                value = args[0].detach().reshape(-1, layer.in_features)[:remaining].cpu()
                captured.append(value)
                count += len(value)

        handle = layer.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                for record in records:
                    model(record["embedding"].to(device), record["hidden"].to(device))
                    if count >= options.max_tokens_per_expert:
                        break
        finally:
            handle.remove()
        packed, stat = quantize_linear(layer, torch.cat(captured), options, device=device)
        payload.update({f"mtp.{name}.{key}": value for key, value in packed.state_dict().items()})
        with torch.no_grad():
            layer.weight.copy_(packed.dequantize_weight().T.to(device=device, dtype=layer.weight.dtype))
        metrics[name] = stat
    model.cpu()
    return payload, metrics


def quantize_linear(layer, inputs, options, *, device):
    from .config import QuantizeConfig, FORMAT
    from .gptq import GPTQ
    from .auto_round import build_round_g_idx
    from ..nn_modules.qlinear.torch import TorchQuantLinear
    rows, cols = layer.weight.shape
    bits = options.block_bits
    if options.algorithm == "gptq":
        config = QuantizeConfig(bits=bits, group_size=64, sym=True, desc_act=False, act_group_aware=False,
                                data_type=options.data_type, format=FORMAT.GPTQ_V2,
                                failsafe=None, offload_to_disk=False)
        solver = GPTQ(layer, qcfg=config)
        solver.quantizer.configure(perchannel=True, bits=bits, sym=True)
        try:
            for chunk in inputs.split(options.batch_tokens):
                solver.add_batch(chunk.to(device=device, dtype=layer.weight.dtype).unsqueeze(0), None)
            quant, scale, zero, groups, _, loss, damp, samples = solver.quantize()
            with torch.no_grad():
                layer.weight.copy_(quant)
            stat = {"algorithm": "gptq", "loss": float(loss), "hessian_tokens": samples}
        finally:
            solver.free()
    else:
        from ..looper.lm_head_auto_round_processor import tune_head_chunk
        config = SimpleNamespace(bits=bits, group_size=64, data_type=options.data_type,
                                 iters=options.iters, lr=1 / options.iters, minmax_lr=1 / options.iters)
        quant, scale, zero, stat = tune_head_chunk(layer.weight, inputs, config,
                                                  {"seed": options.seed, "batch_tokens": options.batch_tokens})
        with torch.no_grad():
            layer.weight.copy_(quant)
        scale = scale.reshape(rows, -1).float().cpu()
        zero = zero.reshape(rows, -1).float().cpu() if torch.is_tensor(zero) else torch.full_like(scale, float(zero))
        groups = build_round_g_idx(cols, 64)
        stat["algorithm"] = "autoround"
    packed = TorchQuantLinear(bits=bits, group_size=64, sym=True, desc_act=False,
                              in_features=cols, out_features=rows)
    packed.pack(layer, scale.cpu(), zero.cpu(), groups.cpu(), workers=1)
    packed.qzero_format(format=2)
    if not torch.isfinite(packed.dequantize_weight()).all():
        raise ValueError("Dense MTP quantization produced non-finite weights")
    return packed, stat
