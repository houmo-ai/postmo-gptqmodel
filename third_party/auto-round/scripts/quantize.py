"""Qwen3.5 quantization script with 4 operational modes.

Modes:
  1. llm-only       : Only quantize LLM (W4G64, no rotation)
  2. rotate-merger   : Rotate LLM + vision merger output, quantize LLM (W4G64)
  3. rotate-full     : Rotate LLM + full vision encoder, quantize LLM (W4G64)
  4. rotate-quant-all: Rotate LLM + full vision, quantize vision W8G64 + LLM W4G64

Flags:
  --deterministic    Force CuBLAS and attention determinism for reproducible results.
  --two_stage        (Mode 4 only) Two-stage calibration: LLM with pile-10k,
                     vision with llava_conv_58k. Avoids the dataset switch
                     that degrades LLM calibration quality.

Usage:
    python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \\
        --mode llm-only --output_dir ./output/mode1-llm-only

    python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \\
        --mode rotate-full --output_dir ./output/mode3-rotate-full --deterministic

    python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \\
        --mode rotate-quant-all --output_dir ./output/mode4-twostage --two_stage
"""

import argparse
import os
import sys
import time

# Set CUBLAS deterministic config BEFORE importing torch (must happen before CUDA init)
if "--deterministic" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 AutoRound quantization (4 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (Qwen3.5 VLM)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["llm-only", "rotate-merger", "rotate-full", "rotate-quant-all"],
                        help="Quantization mode")
    parser.add_argument("--device", type=str, default="cuda:0")

    # LLM quantization params
    parser.add_argument("--llm_bits", type=int, default=4)
    parser.add_argument("--llm_group_size", type=int, default=64)
    # Vision quantization params (mode 4 only)
    parser.add_argument("--vision_bits", type=int, default=8)
    parser.add_argument("--vision_group_size", type=int, default=64)

    parser.add_argument("--sym", action="store_true", default=False,
                        help="Use symmetric quantization (default: asymmetric)")
    parser.add_argument("--iters", type=int, default=200, help="SignSGD iterations (0=RTN)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulate_steps", type=int, default=1,
                        help="Gradient accumulation steps for AutoRound calibration")
    parser.add_argument("--low_gpu_mem_usage", action="store_true", default=False,
                        help="Enable AutoRound low GPU memory mode")
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for rotation matrices and AutoRound")
    parser.add_argument("--format", type=str, default="auto_gptq",
                        help="Export format: auto_gptq (GPTQModel compat), auto_round, etc.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Floating dtype used when loading/saving the pre-quant model")
    parser.add_argument("--save_rotated", type=str, default=None,
                        help="Save rotated (pre-quant) model for debugging")
    parser.add_argument("--only_save_rotated", action="store_true", default=False,
                        help="Exit after saving the rotated pre-quant model; requires --save_rotated")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable deterministic CuBLAS and attention (reproducible results)")
    parser.add_argument("--two_stage", action="store_true", default=False,
                        help="(Mode 4) Two-stage calibration: LLM with pile-10k, vision with llava_conv_58k")

    return parser.parse_args()


# ─── Rotation helpers (adapted from GPTQModel rotation.py) ────────────────────

def _get_hadamard_matrix(dim, device):
    """Generate normalized Hadamard matrix of given dimension."""
    if dim & (dim - 1) == 0:
        Q = torch.tensor([[1.0]], device=device)
        while Q.shape[0] < dim:
            Q = torch.cat([
                torch.cat([Q, Q], dim=1),
                torch.cat([Q, -Q], dim=1),
            ], dim=0) / (2 ** 0.5)
        return Q.to(torch.float64)
    M = torch.randn(dim, dim, device=device, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q


def _get_norm_scale(layernorm):
    """Get multiplicative scale from LayerNorm/RMSNorm."""
    weight = layernorm.weight.data.double()
    if layernorm.__class__.__name__ in ("Qwen3_5RMSNorm",):
        return 1.0 + weight
    return weight


def _fuse_norm_into_linears(layernorm, linear_layers):
    """Absorb norm weights into adjacent linear layers."""
    scale = _get_norm_scale(layernorm)
    for linear in linear_layers:
        dtype = linear.weight.dtype
        W = linear.weight.data.double()
        linear.weight.data = (W * scale).to(dtype)
        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.double() + W @ layernorm.bias.double()).to(dtype)


def _reset_norm(layernorm):
    """Reset norm to identity after fusion."""
    if layernorm.__class__.__name__ in ("Qwen3_5RMSNorm",):
        layernorm.weight.data.zero_()
    else:
        layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, 'bias') and layernorm.bias is not None:
        layernorm.bias.data.zero_()


def _bake_mean_into_linear(linear):
    """Subtract mean from linear weight columns (LayerNorm mean removal)."""
    dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(dtype)


def _bake_mean_into_conv(conv):
    """Subtract per-output-channel mean from conv weight (for LayerNorm fusion)."""
    dtype = conv.weight.dtype
    W = conv.weight.data.double()
    W_flat = W.reshape(W.shape[0], -1)  # [out_c, in_c*kH*kW]
    W_flat = W_flat - W_flat.mean(dim=-1, keepdim=True)
    conv.weight.data = W_flat.reshape(W.shape).to(dtype)


def _rotate_weight(W, Q, device):
    """Apply rotation: W_new = W @ Q"""
    dtype = W.dtype
    return (W.to(device=device, dtype=torch.float64) @ Q).to(dtype=dtype).contiguous()


def _rotate_weight_out(W, Q, device):
    """Apply rotation: W_new = Q^T @ W"""
    dtype = W.dtype
    return (Q.T @ W.to(device=device, dtype=torch.float64)).to(dtype=dtype).contiguous()


def _get_visual(model):
    """Get visual module from Qwen3.5 VLM model."""
    if hasattr(model, 'visual'):
        return model.visual
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        return model.model.visual
    raise AttributeError(f"Cannot find visual module in {type(model).__name__}")


def _get_llm_parts(model):
    """Get LLM layers, embed_tokens, lm_head from model."""
    # Qwen3.5 VLM: model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        return lm.layers, lm.embed_tokens, lm_head, lm.norm
    # Older VLM: model.language_model.model.layers
    if hasattr(model, 'language_model'):
        lm = model.language_model
        return lm.model.layers, lm.model.embed_tokens, lm.lm_head, lm.model.norm
    # Pure CausalLM: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, model.model.embed_tokens, model.lm_head, model.model.norm
    raise ValueError(f"Cannot find LLM layers in {type(model).__name__}")


# ─── LLM Rotation ────────────────────────────────────────────────────────────

def fuse_llm_norms(model, device):
    """Fuse all LLM RMSNorm weights into adjacent linear layers."""
    layers, embed, lm_head, final_norm = _get_llm_parts(model)

    _fuse_norm_into_linears(final_norm, [lm_head])
    _reset_norm(final_norm)

    for layer in layers:
        post_ln = layer.post_attention_layernorm
        input_ln = layer.input_layernorm

        _fuse_norm_into_linears(post_ln, [layer.mlp.gate_proj, layer.mlp.up_proj])

        attn_inputs = []
        if hasattr(layer, "self_attn"):
            attn_inputs.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        if hasattr(layer, "linear_attn"):
            attn_inputs.extend([
                layer.linear_attn.in_proj_qkv,
                layer.linear_attn.in_proj_z,
                layer.linear_attn.in_proj_b,
                layer.linear_attn.in_proj_a,
            ])
        if attn_inputs:
            _fuse_norm_into_linears(input_ln, attn_inputs)

        _reset_norm(post_ln)
        _reset_norm(input_ln)


def rotate_llm(model, Q, device):
    """Apply Hadamard rotation to LLM layers."""
    layers, embed, lm_head, _ = _get_llm_parts(model)

    embed.weight.data = _rotate_weight(embed.weight.data, Q, device)
    lm_head.weight.data = _rotate_weight(lm_head.weight.data, Q, device)

    for i, layer in enumerate(layers):
        print(f"  Rotating LLM layer {i}/{len(layers)}", end="\r")

        if hasattr(layer, "self_attn"):
            for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
                proj.weight.data = _rotate_weight(proj.weight.data, Q, device)
            layer.self_attn.o_proj.weight.data = _rotate_weight_out(
                layer.self_attn.o_proj.weight.data, Q, device)

        if hasattr(layer, "linear_attn"):
            for proj in [layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_z,
                         layer.linear_attn.in_proj_b, layer.linear_attn.in_proj_a]:
                proj.weight.data = _rotate_weight(proj.weight.data, Q, device)
            layer.linear_attn.out_proj.weight.data = _rotate_weight_out(
                layer.linear_attn.out_proj.weight.data, Q, device)

        layer.mlp.gate_proj.weight.data = _rotate_weight(layer.mlp.gate_proj.weight.data, Q, device)
        layer.mlp.up_proj.weight.data = _rotate_weight(layer.mlp.up_proj.weight.data, Q, device)
        layer.mlp.down_proj.weight.data = _rotate_weight_out(
            layer.mlp.down_proj.weight.data, Q, device)

    print(f"  Rotated {len(layers)} LLM layers")


# ─── Vision Rotation ──────────────────────────────────────────────────────────

def fuse_vision_norms(model, device):
    """Fuse vision encoder LayerNorm weights into linear layers."""
    visual = _get_visual(model)

    _bake_mean_into_conv(visual.patch_embed.proj)

    if hasattr(visual, 'pos_embed'):
        W = visual.pos_embed.weight.data.double()
        visual.pos_embed.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(visual.pos_embed.weight.dtype)

    for block in visual.blocks:
        _fuse_ln_linear(block.norm1, [block.attn.qkv])
        _reset_norm(block.norm1)
        _bake_mean_into_linear(block.attn.proj)
        _fuse_ln_linear(block.norm2, [block.mlp.linear_fc1])
        _reset_norm(block.norm2)
        _bake_mean_into_linear(block.mlp.linear_fc2)

    # Merger norm → fc1: fc1.in_features = spatial_merge² × hidden_size, tile scale
    merger_norm = visual.merger.norm
    merger_fc1 = visual.merger.linear_fc1
    _fuse_ln_linear(merger_norm, [merger_fc1], tile_factor=merger_fc1.in_features // merger_norm.weight.shape[0])
    _reset_norm(merger_norm)


def _fuse_ln_linear(layernorm, linear_layers, tile_factor=1):
    """Fuse LayerNorm into linear layers (standard scale=weight form).
    
    tile_factor: repeat scale/bias for merged spatial tokens (e.g. 4 for 2x2 merger).
    """
    scale = layernorm.weight.data.double()
    if tile_factor > 1:
        scale = scale.repeat(tile_factor)
    for linear in linear_layers:
        dtype = linear.weight.dtype
        W = linear.weight.data.double()
        linear.weight.data = (W * scale).to(dtype)
        if layernorm.bias is not None:
            bias_data = layernorm.bias.data.double()
            if tile_factor > 1:
                bias_data = bias_data.repeat(tile_factor)
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.double() + W @ bias_data).to(dtype)

def rotate_vision_full(model, Q_vision, device):
    """Rotate full vision encoder (patch_embed + blocks + merger input)."""
    visual = _get_visual(model)

    # Rotate patch embed conv output → rotated hidden space
    dtype = visual.patch_embed.proj.weight.dtype
    W = visual.patch_embed.proj.weight.data.double().to(device)
    W_flat = W.reshape(W.shape[0], -1)  # [out_c, in_c*kH*kW]
    W_flat = Q_vision.T @ W_flat  # output-side rotation
    visual.patch_embed.proj.weight.data = W_flat.reshape(W.shape).to(dtype)
    if visual.patch_embed.proj.bias is not None:
        b = visual.patch_embed.proj.bias.data.double().to(device)
        visual.patch_embed.proj.bias.data = (Q_vision.T @ b).to(dtype)

    # Rotate positional embedding
    if hasattr(visual, 'pos_embed'):
        visual.pos_embed.weight.data = _rotate_weight(visual.pos_embed.weight.data, Q_vision, device)

    # Rotate each vision block
    for i, block in enumerate(visual.blocks):
        print(f"  Rotating vision block {i}/{len(visual.blocks)}", end="\r")
        block.attn.qkv.weight.data = _rotate_weight(block.attn.qkv.weight.data, Q_vision, device)
        block.attn.proj.weight.data = _rotate_weight_out(block.attn.proj.weight.data, Q_vision, device)
        block.mlp.linear_fc1.weight.data = _rotate_weight(block.mlp.linear_fc1.weight.data, Q_vision, device)
        block.mlp.linear_fc2.weight.data = _rotate_weight_out(
            block.mlp.linear_fc2.weight.data, Q_vision, device)

    print(f"  Rotated {len(visual.blocks)} vision blocks")

    # Rotate merger input to match rotated vision output
    _rotate_merger_input(model, Q_vision, device)


def _rotate_merger_input(model, Q_vision, device):
    """Rotate vision merger input side (linear_fc1) to match vision encoder output."""
    visual = _get_visual(model)
    dtype = visual.merger.linear_fc1.weight.dtype
    out_features, in_features = visual.merger.linear_fc1.weight.shape
    q_dim = Q_vision.shape[0]
    W = visual.merger.linear_fc1.weight.data.double().to(device).reshape(out_features, -1, q_dim)
    visual.merger.linear_fc1.weight.data = torch.matmul(W, Q_vision).to(dtype).reshape(
        out_features, in_features).contiguous()


def rotate_merger_output(model, Q_llm, device):
    """Rotate merger output to match rotated LLM input space."""
    visual = _get_visual(model)
    visual.merger.linear_fc2.weight.data = _rotate_weight_out(
        visual.merger.linear_fc2.weight.data, Q_llm, device)


# ─── Vision layer names (for layer_config) ───────────────────────────────────

def get_vision_layer_names(model):
    """Get all linear layer names in the vision encoder for layer_config."""
    visual = _get_visual(model)
    prefix = None
    for name, mod in model.named_modules():
        if mod is visual:
            prefix = name
            break
    if prefix is None:
        return []

    names = []
    for name, mod in visual.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names.append(f"{prefix}.{name}")
    return names


# ─── Two-stage quantization ──────────────────────────────────────────────────

def _run_two_stage(model, tokenizer, processor, args, ar_kwargs):
    """Two-stage Mode 4: LLM calibrated with pile-10k, vision with llava_conv_58k.

    Stage 1: Quantize LLM only (W4G64) using pile-10k calibration data.
    Stage 2: Reload fresh model, re-apply rotations, quantize vision only (W8G64)
             using llava_conv_58k, with LLM layers skipped via ignore_layers.
    Merge:   Combine LLM quantized weights from Stage 1 with vision quantized
             weights from Stage 2 into the final output.

    Note: Model must be reloaded for Stage 2 because Stage 1's quantize() replaces
    LLM linear layers with QuantLinear wrappers that lack forward().
    """
    import json
    import shutil
    from auto_round import AutoRound

    vis_names = get_vision_layer_names(model)
    print(f"\n{'='*60}")
    print(f"TWO-STAGE QUANTIZATION")
    print(f"  Stage 1: LLM W{args.llm_bits}G{args.llm_group_size} with pile-10k")
    print(f"  Stage 2: Vision W{args.vision_bits}G{args.vision_group_size} "
          f"with llava_conv_58k ({len(vis_names)} layers)")
    print(f"{'='*60}")

    stage1_dir = os.path.join(args.output_dir, "_stage1_llm")
    stage2_dir = os.path.join(args.output_dir, "_stage2_vision")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    # ── Stage 1: LLM quantization with pile-10k ──
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
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        dataset="NeelNanda/pile-10k",
        seed=args.seed,
        quant_nontext_module=False,
        **ar_kwargs,
    )

    t1 = time.time()
    ar1.quantize()
    print(f"Stage 1 quantization: {time.time() - t1:.1f}s")

    print(f"Saving Stage 1 to: {stage1_dir}")
    ar1.save_quantized(stage1_dir, format=args.format)
    del ar1
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 2: Reload fresh model, re-apply rotations, quantize vision ──
    print(f"\n{'─'*60}")
    print(f"STAGE 2: Vision quantization (W{args.vision_bits}G{args.vision_group_size}, llava_conv_58k)")
    print(f"  Reloading fresh model and re-applying rotations...")
    print(f"{'─'*60}")

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    if args.deterministic:
        model_kwargs["attn_implementation"] = "eager"

    try:
        from transformers import AutoModelForImageTextToText
        model2 = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel
        model2 = AutoModel.from_pretrained(args.model, **model_kwargs)
    print(f"  Reloaded: {type(model2).__name__}")

    # Re-apply rotations (same as main() rotation block for rotate-quant-all)
    # Fix seed to reproduce identical Q matrices as Stage 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    text_config = getattr(model2.config, 'text_config', model2.config)
    hidden_size = text_config.hidden_size
    Q_llm = _get_hadamard_matrix(hidden_size, args.device)

    print("  Re-fusing LLM norms...")
    fuse_llm_norms(model2, args.device)
    print("  Re-fusing vision norms...")
    fuse_vision_norms(model2, args.device)

    vis_config = getattr(model2.config, 'vision_config', None)
    vis_hidden = vis_config.hidden_size if vis_config else _get_visual(model2).blocks[0].attn.qkv.in_features
    Q_vision = _get_hadamard_matrix(vis_hidden, args.device)

    print("  Re-rotating full vision encoder...")
    rotate_vision_full(model2, Q_vision, device=args.device)
    print("  Re-rotating merger output → LLM space...")
    rotate_merger_output(model2, Q_llm, device=args.device)
    print("  Re-rotating LLM layers...")
    rotate_llm(model2, Q_llm, device=args.device)
    print("  Rotations re-applied.")

    # Build layer_config: vision layers at W8, LLM skipped via ignore_layers
    vis_names2 = get_vision_layer_names(model2)
    vis_layer_config = {
        name: {"bits": args.vision_bits, "group_size": args.vision_group_size}
        for name in vis_names2
    }

    # ignore_layers uses prefix matching: "model.language_model" skips all LLM layers
    ar2 = AutoRound(
        model=model2,
        tokenizer=tokenizer,
        processor=processor,
        bits=args.vision_bits,
        group_size=args.vision_group_size,
        sym=args.sym,
        iters=args.iters,
        seqlen=512,
        nsamples=args.nsamples,
        batch_size=1,
        gradient_accumulate_steps=8,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
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
    del ar2
    del model2
    gc.collect()
    torch.cuda.empty_cache()

    # ── Merge: LLM from Stage 1 + Vision from Stage 2 ──
    print(f"\n{'─'*60}")
    print(f"MERGING Stage 1 (LLM) + Stage 2 (Vision)")
    print(f"{'─'*60}")

    _merge_two_stages(stage1_dir, stage2_dir, args.output_dir)

    print(f"\n✅ Two-stage merge complete → {args.output_dir}")


def _merge_two_stages(stage1_dir, stage2_dir, output_dir):
    """Merge Stage 1 (LLM quantized) + Stage 2 (vision quantized) safetensors."""
    import json
    import glob as glob_mod
    from safetensors.torch import load_file as st_load, save_file as st_save

    os.makedirs(output_dir, exist_ok=True)

    # Load Stage 1 index (LLM quantized + FP vision)
    idx1_path = os.path.join(stage1_dir, "model.safetensors.index.json")
    idx2_path = os.path.join(stage2_dir, "model.safetensors.index.json")

    with open(idx1_path) as f:
        idx1 = json.load(f)
    with open(idx2_path) as f:
        idx2 = json.load(f)

    wmap1 = idx1["weight_map"]
    wmap2 = idx2["weight_map"]

    # Collect all unique shard files from stage 1
    stage1_shards = set(wmap1.values())
    stage2_shards = set(wmap2.values())

    # Load all tensors from both stages
    all_tensors_s1 = {}
    for shard in stage1_shards:
        shard_path = os.path.join(stage1_dir, shard)
        all_tensors_s1.update(st_load(shard_path, device="cpu"))

    all_tensors_s2 = {}
    for shard in stage2_shards:
        shard_path = os.path.join(stage2_dir, shard)
        all_tensors_s2.update(st_load(shard_path, device="cpu"))

    # Merge: vision tensors from Stage 2, everything else from Stage 1
    # Vision tensor names start with "model.visual."
    merged = {}
    vision_from_s2 = 0
    llm_from_s1 = 0

    # Start with all Stage 1 tensors (LLM quantized + FP vision)
    for name, tensor in all_tensors_s1.items():
        merged[name] = tensor
        llm_from_s1 += 1

    # Override vision tensors with Stage 2 (vision quantized)
    for name, tensor in all_tensors_s2.items():
        if "visual" in name:
            merged[name] = tensor
            vision_from_s2 += 1

    # Also add any vision tensors from Stage 2 not in Stage 1
    # (e.g., qweight, qzeros, scales for vision layers)
    for name, tensor in all_tensors_s2.items():
        if "visual" in name and name not in merged:
            merged[name] = tensor
            vision_from_s2 += 1

    print(f"  Merged: {llm_from_s1} tensors from Stage 1, "
          f"{vision_from_s2} vision tensors overridden from Stage 2")
    print(f"  Total merged tensors: {len(merged)}")

    # Save merged tensors as sharded safetensors
    MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

    # Sort tensor names for deterministic sharding
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

    # Write shards and build new index
    new_weight_map = {}
    total_size = 0
    for i, shard_data in enumerate(shards):
        if len(shards) == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        st_save(shard_data, shard_path)
        for name, tensor in shard_data.items():
            new_weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        print(f"  Saved shard: {shard_name} ({len(shard_data)} tensors)")

    # Write merged index
    if len(shards) > 1:
        merged_index = {
            "metadata": {"total_size": total_size},
            "weight_map": new_weight_map,
        }
        idx_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(idx_path, "w") as f:
            json.dump(merged_index, f, indent=2)

    # Merge quantization configs
    qcfg1_path = os.path.join(stage1_dir, "quantize_config.json")
    qcfg2_path = os.path.join(stage2_dir, "quantize_config.json")
    if os.path.exists(qcfg1_path) and os.path.exists(qcfg2_path):
        with open(qcfg1_path) as f:
            qcfg1 = json.load(f)
        with open(qcfg2_path) as f:
            qcfg2 = json.load(f)

        # Use Stage 1 as base (LLM quantization config)
        merged_qcfg = qcfg1.copy()

        # Merge modules_in_block_to_quantize from both stages
        mibq1 = qcfg1.get("modules_in_block_to_quantize", [])
        mibq2 = qcfg2.get("modules_in_block_to_quantize", [])
        # Deduplicate and combine
        all_modules = set()
        for block_modules in mibq1 + mibq2:
            all_modules.update(block_modules)
        if all_modules:
            merged_qcfg["modules_in_block_to_quantize"] = [sorted(all_modules)]

        # Note in config that this is two-stage
        merged_qcfg["two_stage"] = True
        merged_qcfg["stage1_dataset"] = "NeelNanda/pile-10k"
        merged_qcfg["stage2_dataset"] = "liuhaotian/llava_conv_58k"

        with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
            json.dump(merged_qcfg, f, indent=2)

    # Copy config.json and other non-safetensor files from Stage 1
    for fname in os.listdir(stage1_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        if fname == "quantize_config.json":
            continue  # already merged above
        src = os.path.join(stage1_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    print(f"  Merged quantization config written")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.time()

    need_rotation = args.mode in ("rotate-merger", "rotate-full", "rotate-quant-all")
    need_full_vision_rotation = args.mode in ("rotate-full", "rotate-quant-all")
    need_vision_quant = args.mode == "rotate-quant-all"

    if args.two_stage and args.mode != "rotate-quant-all":
        print("ERROR: --two_stage only works with --mode rotate-quant-all")
        sys.exit(1)

    if args.only_save_rotated and not args.save_rotated:
        print("ERROR: --only_save_rotated requires --save_rotated")
        sys.exit(1)

    if args.only_save_rotated and not need_rotation:
        print("ERROR: --only_save_rotated requires a rotation mode")
        sys.exit(1)

    # ── Deterministic mode ──
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[deterministic] CuBLAS workspace config set, "
              "flash/mem_efficient SDP disabled, cuDNN deterministic")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"LLM: W{args.llm_bits}G{args.llm_group_size}")
    if need_vision_quant:
        print(f"Vision: W{args.vision_bits}G{args.vision_group_size}")
    print(f"Torch dtype: {args.torch_dtype}")
    if args.two_stage:
        print(f"Two-stage: LLM with pile-10k, vision with llava_conv_58k")
    if args.deterministic:
        print(f"Deterministic: ON")
    print(f"Format: {args.format}")
    print(f"{'='*60}")

    # ── Always load full VLM model ──
    from transformers import AutoTokenizer, AutoProcessor
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load processor for MLLM path (needed by MLLMCompressor)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        pass

    # Always load to CPU; AutoRound handles GPU placement during optimization.
    # device_map="auto" causes device mismatch errors during caching for VLMs.
    load_device = "cpu"
    torch_dtype = torch.float16 if args.torch_dtype == "fp16" else torch.bfloat16
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=load_device,
        trust_remote_code=True,
    )
    if args.deterministic:
        model_kwargs["attn_implementation"] = "eager"

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    print(f"Loaded: {type(model).__name__}")

    # ── Rotation ──
    if need_rotation:
        print(f"\n{'='*60}")
        print("Applying rotation...")
        print(f"{'='*60}")
        t0 = time.time()

        # Fix seed before generating rotation matrices so Q_llm/Q_vision are
        # identical across runs (dim 5120/1152 are not power-of-2, so
        # _get_hadamard_matrix falls back to torch.randn + QR).
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        text_config = getattr(model.config, 'text_config', model.config)
        hidden_size = text_config.hidden_size
        Q_llm = _get_hadamard_matrix(hidden_size, args.device)

        # Step 1: Fuse LLM norms
        print("  Fusing LLM norms...")
        fuse_llm_norms(model, args.device)

        # Step 2: Full vision rotation (modes 3, 4)
        if need_full_vision_rotation:
            print("  Fusing vision norms...")
            fuse_vision_norms(model, args.device)

            vis_config = getattr(model.config, 'vision_config', None)
            vis_hidden = vis_config.hidden_size if vis_config else _get_visual(model).blocks[0].attn.qkv.in_features
            Q_vision = _get_hadamard_matrix(vis_hidden, args.device)

            print("  Rotating full vision encoder...")
            rotate_vision_full(model, Q_vision, device=args.device)

        # Step 3: Rotate merger output → LLM space (modes 2, 3, 4)
        print("  Rotating merger output → LLM space...")
        rotate_merger_output(model, Q_llm, device=args.device)

        # Step 4: Rotate LLM
        print("  Rotating LLM layers...")
        rotate_llm(model, Q_llm, device=args.device)

        rot_time = time.time() - t0
        print(f"Rotation completed in {rot_time:.1f}s")

        if args.save_rotated:
            print(f"Saving rotated model to: {args.save_rotated}")
            model.save_pretrained(args.save_rotated, max_shard_size="5GB")
            tokenizer.save_pretrained(args.save_rotated)
            if processor is not None:
                processor.save_pretrained(args.save_rotated)

            if args.only_save_rotated:
                total_time = time.time() - total_start
                print(f"\n{'='*60}")
                print("Done! Rotated pre-quant model saved")
                print(f"   Output: {args.save_rotated}")
                print(f"   Mode: {args.mode}")
                print(f"   Torch dtype: {args.torch_dtype}")
                print(f"   Time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
                print(f"{'='*60}")
                return

    # ── Build layer_config for mode 4 (vision W8) ──
    layer_config = None
    quant_nontext = False
    if need_vision_quant and not args.two_stage:
        # Single-stage: quantize both LLM and vision in one pass
        quant_nontext = True
        vis_names = get_vision_layer_names(model)
        layer_config = {
            name: {"bits": args.vision_bits, "group_size": args.vision_group_size}
            for name in vis_names
        }
        print(f"\nVision quantization: {len(vis_names)} layers at W{args.vision_bits}G{args.vision_group_size}")

    from auto_round import AutoRound

    ar_kwargs = {}
    if args.deterministic:
        ar_kwargs["enable_deterministic_algorithms"] = True

    if args.two_stage and need_vision_quant:
        # ── Two-stage quantization ──
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
            iters=args.iters,
            seqlen=args.seqlen,
            nsamples=args.nsamples,
            batch_size=args.batch_size,
            gradient_accumulate_steps=args.gradient_accumulate_steps,
            low_gpu_mem_usage=args.low_gpu_mem_usage,
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
