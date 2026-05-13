"""Export AutoRound quantized model for xh2modelzoo deployment.

Loads an AutoRound (or GPTQ) quantized model via transformers,
dequantizes weights, and saves both:
  1. Dequantized FP16 weights (for xhquant ONNX graph construction)
  2. Quantization parameters (qweight, scales, zeros) for reference

Usage:
    python scripts/export_xh2.py \
        --quantized_model ./output/qwen35-w4g64-autoround \
        --output_dir ./output/qwen35-xh2-export \
        --dtype bf16
"""

import argparse
import os
import time

import torch


def _patch_backends():
    """Remove backends requiring GLIBC_2.32 (system has 2.31). Must run first."""
    try:
        from auto_round.inference.backend import BackendInfos
        to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
        for k in to_remove:
            del BackendInfos[k]
        if to_remove:
            print(f"Patched: removed {len(to_remove)} incompatible backends")
    except Exception as e:
        print(f"Backend patch skipped: {e}")


# Patch BEFORE importing transformers (which triggers backend discovery)
_patch_backends()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Export AutoRound model for xh2modelzoo")
    parser.add_argument("--quantized_model", type=str, required=True,
                        help="Path to AutoRound/GPTQ quantized model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for exported model")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="Compute dtype for dequantized weights")
    parser.add_argument("--save_quant_params", action="store_true", default=True,
                        help="Also save quantization parameters (qweight, scales, zeros)")
    return parser.parse_args()


def detect_quant_method(model_path):
    """Detect quantization method from model config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, 'quantization_config'):
        qconfig = config.quantization_config
        if isinstance(qconfig, dict):
            return qconfig.get('quant_method', 'unknown')
        return getattr(qconfig, 'quant_method', 'unknown')
    return None


def extract_quant_params(model):
    """Extract quantization parameters from all quantized layers.

    Returns:
        dict: {layer_name: {qweight, scales, zeros, bits, group_size}}
    """
    quant_params = {}

    for name, module in model.named_modules():
        if hasattr(module, 'qweight'):
            params = {
                'qweight': module.qweight.data.cpu(),
                'scales': module.scales.data.cpu() if hasattr(module, 'scales') else None,
                'bits': getattr(module, 'bits', 4),
                'group_size': getattr(module, 'group_size', 128),
            }
            if hasattr(module, 'qzeros'):
                params['zeros'] = module.qzeros.data.cpu()
            if hasattr(module, 'g_idx'):
                params['g_idx'] = module.g_idx.data.cpu()
            quant_params[name] = params

    return quant_params


def dequantize_model(model):
    """Dequantize all quantized layers in-place (GPTQ/auto_gptq packing format).

    Supports auto_round:tritonv2 QuantLinear with qweight/scales/qzeros attributes.
    After this, quantized Linear layers are replaced with standard nn.Linear.
    """
    import torch.nn as nn

    dequantized_count = 0
    replacements = {}

    for name, module in model.named_modules():
        if not (hasattr(module, 'qweight') and hasattr(module, 'scales')):
            continue

        qweight = module.qweight  # (in*bits//32, out) packed int32
        scales = module.scales    # (num_groups, out)
        bits = getattr(module, 'bits', 4)
        group_size = getattr(module, 'group_size', 64)
        bias = module.bias if hasattr(module, 'bias') else None
        has_zeros = hasattr(module, 'qzeros') and module.qzeros is not None

        # Unpack qweight: (in*bits//32, out) int32 → (in, out) int
        pack_factor = 32 // bits
        wf = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device).unsqueeze(0)
        # qweight shape: (rows, out) where rows = in_features * bits // 32
        weight_unpacked = torch.bitwise_right_shift(
            qweight.unsqueeze(1).expand(-1, pack_factor, -1), wf.unsqueeze(-1)
        ).to(torch.int8)
        weight_unpacked = torch.bitwise_and(weight_unpacked, (1 << bits) - 1)
        # Reshape to (in_features, out_features)
        in_features = qweight.shape[0] * pack_factor
        out_features = qweight.shape[1]
        weight_unpacked = weight_unpacked.reshape(in_features, out_features)  # (in, out)

        # Build g_idx (group index for each input feature)
        if hasattr(module, 'g_idx') and module.g_idx is not None:
            g_idx = module.g_idx
        else:
            g_idx = torch.arange(in_features, device=qweight.device) // group_size

        # Unpack qzeros or use symmetric zero
        if has_zeros:
            qzeros = module.qzeros  # (num_groups, out // pack_factor) packed int32
            zeros_unpacked = torch.bitwise_right_shift(
                qzeros.unsqueeze(2).expand(-1, -1, pack_factor), wf.unsqueeze(0)
            ).to(torch.int8)
            zeros_unpacked = torch.bitwise_and(zeros_unpacked, (1 << bits) - 1)
            zeros_unpacked = zeros_unpacked.reshape(scales.shape[0], out_features)
            # auto_round uses gptq_v2 format (no +1 offset)
            zeros_unpacked = zeros_unpacked.to(torch.float32)
        else:
            zeros_unpacked = torch.zeros(scales.shape, dtype=torch.float32, device=qweight.device)

        # Dequantize: weight_fp = (w - z) * scale, grouped
        # g_idx maps each row (in_feature) to its group
        w_float = weight_unpacked.to(torch.float32)
        scale_per_row = scales[g_idx].to(torch.float32)   # (in, out)
        zero_per_row = zeros_unpacked[g_idx].to(torch.float32)  # (in, out)
        weight_fp = (w_float - zero_per_row) * scale_per_row  # (in, out)

        # Transpose to (out, in) for nn.Linear
        weight_fp = weight_fp.T.contiguous()

        # Create replacement nn.Linear
        new_linear = nn.Linear(in_features, out_features, bias=(bias is not None))
        new_linear.weight = nn.Parameter(weight_fp.to(scales.dtype))
        if bias is not None:
            new_linear.bias = nn.Parameter(bias.clone())
        replacements[name] = new_linear
        dequantized_count += 1

    # Apply replacements (can't modify dict while iterating named_modules)
    for name, new_layer in replacements.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer.to(next(model.parameters()).device))

    print(f"Dequantized {dequantized_count} layers")
    return model


def save_dequantized_model(model, tokenizer, output_dir, dtype):
    """Save the dequantized model as a standard HF model."""
    compute_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    # Convert all parameters to target dtype
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float64):
            param.data = param.data.to(compute_dtype)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    # Remove quantization config from saved config
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        if 'quantization_config' in config:
            # Keep as metadata but mark as dequantized
            config['_original_quantization_config'] = config.pop('quantization_config')
            config['_dequantized'] = True
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Dequantized model saved to: {output_dir}")


def save_quant_params(quant_params, output_dir):
    """Save quantization parameters for reference and re-quantization."""
    params_dir = os.path.join(output_dir, "quant_params")
    os.makedirs(params_dir, exist_ok=True)

    for layer_name, params in quant_params.items():
        safe_name = layer_name.replace('.', '_')
        layer_dir = os.path.join(params_dir, safe_name)
        os.makedirs(layer_dir, exist_ok=True)

        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                torch.save(value, os.path.join(layer_dir, f"{key}.pt"))
            else:
                import json
                with open(os.path.join(layer_dir, "meta.json"), 'a') as f:
                    json.dump({key: value}, f)
                    f.write('\n')

    # Save summary
    summary = {
        'num_quantized_layers': len(quant_params),
        'layers': list(quant_params.keys()),
        'bits': list(set(p.get('bits', 4) for p in quant_params.values())),
        'group_sizes': list(set(p.get('group_size', 128) for p in quant_params.values())),
    }
    import json
    with open(os.path.join(params_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Quantization parameters saved to: {params_dir}")
    print(f"  {len(quant_params)} quantized layers")


def main():
    args = parse_args()
    t0 = time.time()

    # ── Detect quantization method ──
    quant_method = detect_quant_method(args.quantized_model)
    print(f"Quantization method: {quant_method}")

    # ── Load quantized model ──
    print(f"\nLoading quantized model: {args.quantized_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.quantized_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.quantized_model,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model class: {type(model).__name__}")

    # ── Extract quantization parameters (before dequantization) ──
    if args.save_quant_params:
        print("\nExtracting quantization parameters...")
        quant_params = extract_quant_params(model)
        if quant_params:
            save_quant_params(quant_params, args.output_dir)
        else:
            print("No quantized layers found (model may already be dequantized)")

    # ── Dequantize ──
    print("\nDequantizing model...")
    model = dequantize_model(model)

    # ── Save dequantized model ──
    print(f"\nSaving dequantized model (dtype={args.dtype})...")
    dequant_dir = os.path.join(args.output_dir, "dequantized_model")
    save_dequantized_model(model, tokenizer, dequant_dir, args.dtype)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Export complete! Total time: {total_time:.1f}s")
    print(f"Output directory: {args.output_dir}")
    print(f"  dequantized_model/ - FP16/BF16 model for xhquant graph conversion")
    if args.save_quant_params:
        print(f"  quant_params/      - Original quantization parameters")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
