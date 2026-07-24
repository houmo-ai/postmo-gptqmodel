"""Gemma4 dense/Unified quantization script with 3 operational modes.

Modes:
  1. llm-only       : Only quantize LLM (W4G64, no rotation)
  2. rotate-merger   : Rotate LLM + vision projection output, quantize LLM (W4G64)
  3. rotate-full     : Rotate LLM + full vision encoder + projection, quantize LLM (W4G64)

Key differences from Qwen3.5 script:
  - Sandwich norms: 4 norms per layer (input, post_attn, pre_ff, post_ff)
  - Post-norms absorbed into output projections (approximate fusion)
  - tie_word_embeddings=True: un-tied before rotation for correctness
  - Vision projections wrapped in .linear container (ClippedLinear)
  - Single-layer vision projection (embed_vision.embedding_projection)
  - layer_scalar per layer (scalar, commutes with rotation)

Usage:
    python scripts/quantize_gemma4.py --model weights/gemma-4-31B-it \\
        --mode llm-only --output_dir ./output/gemma4-mode1 --device cuda:1

    python scripts/quantize_gemma4.py --model weights/gemma-4-31B-it \\
        --mode rotate-merger --output_dir ./output/gemma4-mode2 --device cuda:2

    python scripts/quantize_gemma4.py --model weights/gemma-4-31B-it \\
        --mode rotate-full --output_dir ./output/gemma4-mode3 --device cuda:5
"""

import argparse
import json
import os
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemma4 AutoRound quantization (3 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (Gemma4 VLM)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["llm-only", "rotate-merger", "rotate-full"],
                        help="Quantization mode")
    parser.add_argument("--device", type=str, default="cuda:0")

    # LLM quantization params
    parser.add_argument("--llm_bits", type=int, default=4)
    parser.add_argument("--llm_group_size", type=int, default=64)

    parser.add_argument("--sym", action="store_true", default=False,
                        help="Use symmetric quantization (default: asymmetric)")
    parser.add_argument("--iters", type=int, default=200, help="SignSGD iterations (0=RTN)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for rotation matrices and AutoRound")
    parser.add_argument("--format", type=str, default="auto_gptq",
                        help="Export format: auto_gptq (GPTQModel compat), auto_round, etc.")
    parser.add_argument("--save_rotated", type=str, default=None,
                        help="Save rotated (pre-quant) model for debugging")

    return parser.parse_args()


def restore_source_config_fields(model_dir, output_dir):
    """Preserve checkpoint-only multimodal config fields after HF save."""

    source_path = os.path.join(model_dir, "config.json")
    output_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(source_path) or not os.path.isfile(output_path):
        return

    try:
        with open(source_path, encoding="utf-8") as file:
            source_config = json.load(file)
        with open(output_path, encoding="utf-8") as file:
            output_config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(source_config, dict) or not isinstance(output_config, dict):
        return

    def merge(source, output):
        merged = dict(source)
        for key, output_value in output.items():
            source_value = merged.get(key)
            if isinstance(source_value, dict) and isinstance(output_value, dict):
                merged[key] = merge(source_value, output_value)
            else:
                merged[key] = output_value
        return merged

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merge(source_config, output_config), file, ensure_ascii=False, indent=2)
        file.write("\n")


# ─── Rotation helpers ─────────────────────────────────────────────────────────

def _get_hadamard_matrix(dim, device):
    """Generate normalized orthogonal matrix of given dimension.

    For power-of-2 dimensions, uses Hadamard construction.
    Otherwise, falls back to torch.randn + QR decomposition.
    Seed MUST be fixed before calling for reproducibility.
    """
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
    """Get multiplicative scale from RMSNorm.

    Gemma4RMSNorm uses standard form: output = weight * x / rms(x).
    Some norms have with_scale=False (no weight param) → scale is 1.
    """
    if not hasattr(layernorm, 'weight'):
        return None  # No scale to fuse
    return layernorm.weight.data.double()


def _fuse_norm_into_linears(layernorm, linear_layers):
    """Absorb norm weights into adjacent linear layers (input side).

    For pre-norms: pre_norm(x) → linear(pre_norm(x))
    After fusion: linear_fused(x/rms(x)) gives same result.
    If norm has no learnable weight (with_scale=False), nothing to fuse.
    """
    scale = _get_norm_scale(layernorm)
    if scale is None:
        return  # No scale to fuse (with_scale=False)
    for linear in linear_layers:
        dtype = linear.weight.dtype
        W = linear.weight.data.double()
        linear.weight.data = (W * scale).to(dtype)
        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.double() + W @ layernorm.bias.double()).to(dtype)


def _fuse_norm_into_output(layernorm, linear):
    """Absorb post-norm weight into output projection (output side).

    For post-norms applied AFTER a linear layer:
      post_norm(linear(x)) = weight * linear(x) / rms(linear(x))

    We absorb weight into linear's output rows:
      linear_new[i,:] = weight[i] * linear[i,:]

    After fusion, post_norm becomes x/rms(x) (rotation-invariant).
    This is approximate: rms(w*y) ≈ rms(y) when w is close to 1.
    If norm has no weight (with_scale=False), nothing to fuse.
    """
    scale = _get_norm_scale(layernorm)
    if scale is None:
        return  # No scale to fuse
    dtype = linear.weight.dtype
    # Multiply each output row by scale
    linear.weight.data = (linear.weight.data.double() * scale.unsqueeze(1)).to(dtype)
    if linear.bias is not None:
        linear.bias.data = (linear.bias.data.double() * scale).to(dtype)


def _reset_norm(layernorm):
    """Reset norm to identity after fusion.

    Gemma4RMSNorm uses standard form: weight=1 gives identity scale.
    If norm has no weight (with_scale=False), nothing to reset.
    """
    if hasattr(layernorm, 'weight'):
        layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, 'bias') and layernorm.bias is not None:
        layernorm.bias.data.zero_()


def _bake_mean_into_linear(linear):
    """Subtract mean from linear weight columns (for rotation quality).

    Ensures output has zero mean per feature, improving rotation effectiveness.
    """
    dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(dtype)


def _rotate_weight(W, Q, device):
    """Apply input-side rotation: W_new = W @ Q"""
    dtype = W.dtype
    return (W.to(device=device, dtype=torch.float64) @ Q).to(dtype=dtype).contiguous()


def _rotate_weight_out(W, Q, device):
    """Apply output-side rotation: W_new = Q^T @ W"""
    dtype = W.dtype
    return (Q.T @ W.to(device=device, dtype=torch.float64)).to(dtype=dtype).contiguous()


# ─── Gemma4 model accessors ──────────────────────────────────────────────────

def _get_llm_parts(model):
    """Get LLM layers, embed_tokens, lm_head, final_norm from Gemma4 model.

    Structure: model.model.language_model has layers, embed_tokens, norm.
    lm_head is at model.lm_head (tied to embed_tokens by default).
    """
    lm = model.model.language_model
    lm_head = model.lm_head if hasattr(model, 'lm_head') else None
    return lm.layers, lm.embed_tokens, lm_head, lm.norm


def _get_vision_tower(model):
    """Get vision tower from Gemma4 model."""
    return model.model.vision_tower


def _get_vision_projection(model):
    """Get vision-to-LLM projection from Gemma4 model.

    Also returns the pre-projection norm (embedding_pre_projection_norm).
    """
    ev = model.model.embed_vision
    return ev.embedding_projection, ev.embedding_pre_projection_norm


def _get_vision_linear(module):
    """Get the actual nn.Linear from a possibly wrapped module (ClippedLinear)."""
    if hasattr(module, 'linear'):
        return module.linear
    return module


# ─── LLM Rotation ────────────────────────────────────────────────────────────

def _untie_lm_head(model):
    """Un-tie lm_head from embed_tokens for independent rotation.

    Gemma4 has tie_word_embeddings=True. For rotation, lm_head and embed_tokens
    need different transformations (lm_head gets final norm fusion + rotation,
    embed_tokens only gets rotation).
    """
    layers, embed, lm_head, _ = _get_llm_parts(model)
    if lm_head is None:
        return

    is_tied = (lm_head.weight.data_ptr() == embed.weight.data_ptr())
    if not is_tied:
        return

    print("  Un-tying lm_head from embed_tokens...")
    vocab_size, hidden_size = lm_head.weight.shape
    dtype = lm_head.weight.dtype
    device = lm_head.weight.device

    # Create a new, independent weight for lm_head
    new_weight = lm_head.weight.data.clone()
    model.lm_head.weight = torch.nn.Parameter(new_weight)

    model.config.tie_word_embeddings = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.tie_word_embeddings = False

    # Verify un-tied
    assert model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()
    print(f"  Un-tied: lm_head now has independent weight [{vocab_size}×{hidden_size}]")


def fuse_llm_norms(model, device):
    """Fuse all LLM norms into adjacent linear layers.

    Gemma4 has 4 norms per layer:
    - input_layernorm → q_proj, k_proj, v_proj (pre-norm, fuse into input)
    - post_attention_layernorm → o_proj (post-norm, fuse into output, approximate)
    - pre_feedforward_layernorm → gate_proj, up_proj (pre-norm, fuse into input)
    - post_feedforward_layernorm → down_proj (post-norm, fuse into output, approximate)
    """
    layers, embed, lm_head, final_norm = _get_llm_parts(model)

    # Final norm → lm_head (only if un-tied)
    if lm_head is not None:
        is_tied = (lm_head.weight.data_ptr() == embed.weight.data_ptr())
        if not is_tied:
            _fuse_norm_into_linears(final_norm, [lm_head])
    _reset_norm(final_norm)

    for i, layer in enumerate(layers):
        # Collect attention projections (v_proj is None for k_eq_v layers)
        attn_projs = [layer.self_attn.q_proj, layer.self_attn.k_proj]
        if layer.self_attn.v_proj is not None:
            attn_projs.append(layer.self_attn.v_proj)

        # Pre-norm: input_layernorm → q_proj, k_proj[, v_proj]
        _fuse_norm_into_linears(layer.input_layernorm, attn_projs)
        _reset_norm(layer.input_layernorm)

        # Post-norm: post_attention_layernorm → o_proj (output side, approximate)
        _fuse_norm_into_output(layer.post_attention_layernorm, layer.self_attn.o_proj)
        _reset_norm(layer.post_attention_layernorm)

        # Pre-norm: pre_feedforward_layernorm → gate_proj, up_proj
        _fuse_norm_into_linears(layer.pre_feedforward_layernorm,
                                [layer.mlp.gate_proj, layer.mlp.up_proj])
        _reset_norm(layer.pre_feedforward_layernorm)

        # Post-norm: post_feedforward_layernorm → down_proj (output side, approximate)
        _fuse_norm_into_output(layer.post_feedforward_layernorm, layer.mlp.down_proj)
        _reset_norm(layer.post_feedforward_layernorm)

    print(f"  Fused norms for {len(layers)} LLM layers (4 norms each)")


def rotate_llm(model, Q, device):
    """Apply Hadamard rotation to Gemma4 LLM layers.

    Rotates the residual stream: embed_tokens output and all layer
    input/output projections are transformed so that the hidden
    representation is in rotated space: x_rot = x @ Q.
    """
    layers, embed, lm_head, _ = _get_llm_parts(model)

    # Rotate embedding
    embed.weight.data = _rotate_weight(embed.weight.data, Q, device)

    # Rotate lm_head (if un-tied, it's independent; if tied, already rotated with embed)
    if lm_head is not None and lm_head.weight.data_ptr() != embed.weight.data_ptr():
        lm_head.weight.data = _rotate_weight(lm_head.weight.data, Q, device)

    for i, layer in enumerate(layers):
        print(f"  Rotating LLM layer {i}/{len(layers)}", end="\r")

        # Input-side rotation: q, k[, v] projections
        projs = [layer.self_attn.q_proj, layer.self_attn.k_proj]
        if layer.self_attn.v_proj is not None:
            projs.append(layer.self_attn.v_proj)
        for proj in projs:
            proj.weight.data = _rotate_weight(proj.weight.data, Q, device)

        # Output-side rotation: o_proj
        _bake_mean_into_linear(layer.self_attn.o_proj)
        layer.self_attn.o_proj.weight.data = _rotate_weight_out(
            layer.self_attn.o_proj.weight.data, Q, device)

        # MLP input-side: gate_proj, up_proj
        layer.mlp.gate_proj.weight.data = _rotate_weight(
            layer.mlp.gate_proj.weight.data, Q, device)
        layer.mlp.up_proj.weight.data = _rotate_weight(
            layer.mlp.up_proj.weight.data, Q, device)

        # MLP output-side: down_proj
        _bake_mean_into_linear(layer.mlp.down_proj)
        layer.mlp.down_proj.weight.data = _rotate_weight_out(
            layer.mlp.down_proj.weight.data, Q, device)

    print(f"  Rotated {len(layers)} LLM layers          ")


# ─── Vision Rotation ─────────────────────────────────────────────────────────

def fuse_vision_norms(model, device):
    """Fuse vision encoder norms into adjacent linear layers.

    Gemma4 vision encoder has the same sandwich norm structure as LLM:
    4 norms per layer (input, post_attn, pre_ff, post_ff).
    Vision projections are wrapped in .linear containers.
    """
    vision = _get_vision_tower(model)

    for i, layer in enumerate(vision.encoder.layers):
        # Get actual Linear modules (unwrap ClippedLinear containers)
        q_linear = _get_vision_linear(layer.self_attn.q_proj)
        k_linear = _get_vision_linear(layer.self_attn.k_proj)
        v_linear = _get_vision_linear(layer.self_attn.v_proj)
        o_linear = _get_vision_linear(layer.self_attn.o_proj)
        gate_linear = _get_vision_linear(layer.mlp.gate_proj)
        up_linear = _get_vision_linear(layer.mlp.up_proj)
        down_linear = _get_vision_linear(layer.mlp.down_proj)

        # Pre-norm: input_layernorm → q, k, v
        _fuse_norm_into_linears(layer.input_layernorm, [q_linear, k_linear, v_linear])
        _reset_norm(layer.input_layernorm)

        # Post-norm: post_attention_layernorm → o_proj (output side)
        _fuse_norm_into_output(layer.post_attention_layernorm, o_linear)
        _reset_norm(layer.post_attention_layernorm)

        # Pre-norm: pre_feedforward_layernorm → gate, up
        _fuse_norm_into_linears(layer.pre_feedforward_layernorm, [gate_linear, up_linear])
        _reset_norm(layer.pre_feedforward_layernorm)

        # Post-norm: post_feedforward_layernorm → down_proj (output side)
        _fuse_norm_into_output(layer.post_feedforward_layernorm, down_linear)
        _reset_norm(layer.post_feedforward_layernorm)

    print(f"  Fused norms for {len(vision.encoder.layers)} vision layers")


def rotate_vision_full(model, Q_vision, device):
    """Rotate full vision encoder.

    Rotates patch embeddings, all transformer layers, and the
    standardization parameters (std_bias, std_scale) in vision hidden space.
    """
    vision = _get_vision_tower(model)

    # Rotate patch embedding output (Linear: out=[1152, 768])
    # patch_proj produces vision hidden states, rotate output side
    patch_proj = vision.patch_embedder.input_proj
    dtype = patch_proj.weight.dtype
    patch_proj.weight.data = _rotate_weight_out(
        patch_proj.weight.data, Q_vision, device)
    if patch_proj.bias is not None:
        b = patch_proj.bias.data.double().to(device)
        patch_proj.bias.data = (Q_vision.T @ b).to(dtype)

    # Rotate position embedding table: shape [2, num_positions, hidden_size]
    if hasattr(vision.patch_embedder, 'position_embedding_table'):
        pos_table = vision.patch_embedder.position_embedding_table
        if pos_table is not None:
            pos_data = pos_table.data.double().to(device)
            # Rotate last dim: [2, N, H] @ Q → [2, N, H]
            pos_table.data = (pos_data @ Q_vision).to(pos_table.dtype)

    # Handle std_scale and std_bias
    # std_scale/std_bias are applied element-wise in the vision hidden space:
    #   x_vis = patch_proj(image) * std_scale + std_bias
    # After rotation, we need: x_vis_rot = R^T @ (patch_proj(image) * std_scale + std_bias)
    # But patch_proj output is already rotated: patch_rot = R^T @ patch_orig
    # So: x_vis_rot = (R^T @ diag(std_scale) @ R) @ patch_rot + R^T @ std_bias
    # R^T @ diag(std_scale) @ R is a full matrix → absorb into patch_proj weights
    if hasattr(vision, 'std_scale') and vision.std_scale is not None:
        std_scale = vision.std_scale.data.double().to(device).squeeze()
        # Current patch_proj.weight has been output-rotated: W_rot = Q^T @ W_orig
        # We want to apply std_scale BEFORE output rotation:
        # Final = Q^T @ diag(std_scale) @ W_orig = Q^T @ diag(std_scale) @ Q @ W_rot
        scale_matrix = Q_vision.T @ torch.diag(std_scale) @ Q_vision
        W = patch_proj.weight.data.double().to(device)
        patch_proj.weight.data = (scale_matrix @ W).to(dtype)

        if patch_proj.bias is not None:
            # bias was rotated: b_rot = Q^T @ b_orig
            # Apply scale: Q^T @ diag(std_scale) @ b_orig = Q^T @ diag(std_scale) @ Q @ b_rot
            b_rot = patch_proj.bias.data.double().to(device)
            b_rot = scale_matrix @ b_rot
            patch_proj.bias.data = b_rot.to(dtype)

        vision.std_scale.data.fill_(1.0)

    if hasattr(vision, 'std_bias') and vision.std_bias is not None:
        std_bias = vision.std_bias.data.double().to(device).squeeze()
        rotated_bias = Q_vision.T @ std_bias
        if patch_proj.bias is not None:
            patch_proj.bias.data = (patch_proj.bias.data.double().to(device) + rotated_bias).to(dtype)
        else:
            patch_proj.bias = torch.nn.Parameter(rotated_bias.to(dtype))
        vision.std_bias.data.zero_()

    # Rotate each vision encoder layer
    for i, layer in enumerate(vision.encoder.layers):
        print(f"  Rotating vision layer {i}/{len(vision.encoder.layers)}", end="\r")

        q_linear = _get_vision_linear(layer.self_attn.q_proj)
        k_linear = _get_vision_linear(layer.self_attn.k_proj)
        v_linear = _get_vision_linear(layer.self_attn.v_proj)
        o_linear = _get_vision_linear(layer.self_attn.o_proj)
        gate_linear = _get_vision_linear(layer.mlp.gate_proj)
        up_linear = _get_vision_linear(layer.mlp.up_proj)
        down_linear = _get_vision_linear(layer.mlp.down_proj)

        # Input-side: q, k, v
        for proj in [q_linear, k_linear, v_linear]:
            proj.weight.data = _rotate_weight(proj.weight.data, Q_vision, device)

        # Output-side: o_proj
        _bake_mean_into_linear(o_linear)
        o_linear.weight.data = _rotate_weight_out(o_linear.weight.data, Q_vision, device)

        # MLP input-side: gate, up
        gate_linear.weight.data = _rotate_weight(gate_linear.weight.data, Q_vision, device)
        up_linear.weight.data = _rotate_weight(up_linear.weight.data, Q_vision, device)

        # MLP output-side: down
        _bake_mean_into_linear(down_linear)
        down_linear.weight.data = _rotate_weight_out(down_linear.weight.data, Q_vision, device)

    print(f"  Rotated {len(vision.encoder.layers)} vision layers          ")


def rotate_projection_output(model, Q_llm, device):
    """Rotate vision projection output → LLM rotated space.

    Mode 2: projection output enters LLM residual stream (rotated by Q_llm).
    proj_new.weight = Q_llm^T @ proj.weight
    Also fuses the embedding_pre_projection_norm into the projection layer.
    """
    proj, pre_norm = _get_vision_projection(model)
    # Fuse pre-projection norm into projection input
    _fuse_norm_into_linears(pre_norm, [proj])
    _reset_norm(pre_norm)
    # Rotate output
    _bake_mean_into_linear(proj)
    proj.weight.data = _rotate_weight_out(proj.weight.data, Q_llm, device)
    print("  Rotated projection output → LLM space (fused pre-proj norm)")


def rotate_projection_input(model, Q_vision, device):
    """Rotate vision projection input ← vision rotated space.

    Mode 3: projection input comes from rotated vision encoder.
    Pre-projection norm was already fused (in rotate_projection_output).
    proj_new.weight = proj.weight @ Q_vision
    """
    proj, _ = _get_vision_projection(model)
    proj.weight.data = _rotate_weight(proj.weight.data, Q_vision, device)
    print("  Rotated projection input ← vision space")


# ─── Vision layer names (for layer_config) ───────────────────────────────────

def get_vision_layer_names(model):
    """Get all linear layer names in the vision encoder for layer_config."""
    vision = _get_vision_tower(model)
    prefix = None
    for name, mod in model.named_modules():
        if mod is vision:
            prefix = name
            break
    if prefix is None:
        return []

    names = []
    for name, mod in vision.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names.append(f"{prefix}.{name}")
    return names


# ─── Position embeddings fix for AutoRound block-wise processing ──────────────

def patch_position_embeddings(model, seqlen, device):
    """Monkey-patch decoder layers to use correct position_embeddings per layer type.

    Gemma4 computes different position_embeddings for sliding vs full attention.
    AutoRound caches inputs from layer 0 and reuses for all layers, but layer 0's
    position_embeddings may not match other layers' requirements. Fix: pre-compute
    both types and have each layer use the correct one based on its type.
    """
    import inspect
    import types

    lm = model.model.language_model
    text_config = getattr(model.config, 'text_config', model.config)

    # Pre-compute position embeddings for all layer types (on CPU, model is on CPU)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    dummy_hidden = torch.zeros(1, seqlen, text_config.hidden_size, dtype=torch.bfloat16)

    pos_emb_cache = {}
    for layer_type in set(text_config.layer_types):
        cos, sin = lm.rotary_emb(dummy_hidden, position_ids, layer_type)
        pos_emb_cache[layer_type] = (cos, sin)
        print(f"  Pre-computed position_embeddings for '{layer_type}': "
              f"cos={cos.shape}, sin={sin.shape}")

    # Determine layer type for each layer
    layer_types = text_config.layer_types

    # Original forward reference
    original_forward = type(lm.layers[0]).forward
    forward_parameters = inspect.signature(original_forward).parameters
    supports_per_layer_input = "per_layer_input" in forward_parameters
    supports_shared_kv_states = "shared_kv_states" in forward_parameters

    def make_patched_forward(layer_type_name, orig_fwd):
        cached_pos = pos_emb_cache[layer_type_name]

        def patched_forward(self, hidden_states, per_layer_input=None,
                            position_embeddings=None, attention_mask=None,
                            position_ids=None, past_key_values=None, **kwargs):
            # Replace position_embeddings with the correct type for this layer
            correct_pos = (cached_pos[0].to(hidden_states.device),
                           cached_pos[1].to(hidden_states.device))
            shared_kv_states = kwargs.pop("shared_kv_states", None)
            call_kwargs = dict(
                position_embeddings=correct_pos,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            if supports_per_layer_input:
                call_kwargs["per_layer_input"] = per_layer_input
            if supports_shared_kv_states:
                # Unified marks the last non-sharing layer of each attention
                # type as a KV producer even when num_kv_shared_layers=0.
                # Top-level HF forward always supplies this dictionary, but
                # AutoRound replays decoder blocks directly and otherwise
                # reaches those final layers with None.
                call_kwargs["shared_kv_states"] = (
                    shared_kv_states if shared_kv_states is not None else {}
                )
            return orig_fwd(self, hidden_states, **call_kwargs)
        return patched_forward

    for i, layer in enumerate(lm.layers):
        lt = layer_types[i]
        layer.forward = types.MethodType(make_patched_forward(lt, original_forward), layer)

    print(f"  Patched {len(lm.layers)} layers with correct position_embeddings")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.time()

    need_rotation = args.mode in ("rotate-merger", "rotate-full")
    need_full_vision_rotation = args.mode == "rotate-full"

    print(f"\n{'='*60}")
    print("Gemma4 AutoRound Quantization")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"LLM: W{args.llm_bits}G{args.llm_group_size}")
    print(f"Sym: {args.sym}")
    print(f"Format: {args.format}")
    print(f"Device: {args.device}")
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

    # Load to CPU; AutoRound handles GPU placement during optimization.
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    print(f"Loaded: {type(model).__name__}")

    # Print model info
    text_config = getattr(model.config, 'text_config', model.config)
    print(f"  LLM: {text_config.num_hidden_layers} layers, hidden={text_config.hidden_size}")
    vis_config = getattr(model.config, 'vision_config', None)
    if vis_config:
        vision_layers = getattr(vis_config, "num_hidden_layers", None)
        if vision_layers is None:
            print(
                "  Vision: encoder-free, "
                f"soft_tokens={getattr(vis_config, 'num_soft_tokens', 'unknown')}"
            )
        else:
            print(f"  Vision: {vision_layers} layers, hidden={vis_config.hidden_size}")
    print(f"  tie_word_embeddings: {model.config.tie_word_embeddings}")

    # ── Rotation ──
    if need_rotation:
        print(f"\n{'='*60}")
        print("Applying rotation...")
        print(f"{'='*60}")
        t0 = time.time()

        # Fix seed before generating rotation matrices
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        hidden_size = text_config.hidden_size
        Q_llm = _get_hadamard_matrix(hidden_size, args.device)
        print(f"  Generated Q_llm [{hidden_size}×{hidden_size}]")

        # Step 0: Un-tie lm_head from embed_tokens
        _untie_lm_head(model)

        # Step 1: Fuse LLM norms
        print("  Fusing LLM norms (4 per layer)...")
        fuse_llm_norms(model, args.device)

        # Step 2: Full vision rotation (Mode 3)
        if need_full_vision_rotation:
            print("  Fusing vision norms...")
            fuse_vision_norms(model, args.device)

            vis_hidden = vis_config.hidden_size if vis_config else 1152
            Q_vision = _get_hadamard_matrix(vis_hidden, args.device)
            print(f"  Generated Q_vision [{vis_hidden}×{vis_hidden}]")

            print("  Rotating full vision encoder...")
            rotate_vision_full(model, Q_vision, device=args.device)

            print("  Rotating projection input ← vision space...")
            rotate_projection_input(model, Q_vision, device=args.device)

        # Step 3: Rotate projection output → LLM space (Modes 2, 3)
        print("  Rotating projection output → LLM space...")
        rotate_projection_output(model, Q_llm, device=args.device)

        # Step 4: Rotate LLM
        print("  Rotating LLM layers...")
        rotate_llm(model, Q_llm, device=args.device)

        rot_time = time.time() - t0
        print(f"Rotation completed in {rot_time:.1f}s")

        if args.save_rotated:
            print(f"Saving rotated model to: {args.save_rotated}")
            model.save_pretrained(args.save_rotated, max_shard_size="5GB")

    # ── Patch position embeddings ──
    print("\nPatching position embeddings for block-wise processing...")
    patch_position_embeddings(model, args.seqlen, args.device)

    # ── Quantize ──
    print(f"\n{'='*60}")
    print("Starting AutoRound quantization...")
    print(f"{'='*60}")

    from auto_round import AutoRound

    ar_kwargs = {}
    layer_config = {}
    if getattr(text_config, "hidden_size_per_layer_input", 0):
        # E-series PLE gate/projection linears are not represented by the
        # Gemma3 GPTQModel loader used for Gemma4 artifacts. Keep them as FP
        # weights, matching the GPTQ recipe and avoiding qweight sidecars that
        # are ignored on reload/export.
        layer_config.update({
            "per_layer_input_gate": {"bits": 16, "act_bits": 16},
            "per_layer_projection": {"bits": 16, "act_bits": 16},
        })
        print("  Keeping per-layer input gate/projection in FP16 for GPTQModel compatibility")

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
        dataset=args.dataset,
        seed=args.seed,
        quant_nontext_module=False,
        layer_config=layer_config or None,
        **ar_kwargs,
    )

    t1 = time.time()
    ar.quantize()
    quant_time = time.time() - t1
    print(f"Quantization completed in {quant_time:.1f}s ({quant_time/3600:.2f}h)")

    # ── Save ──
    print(f"\nSaving to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    ar.save_quantized(args.output_dir, format=args.format)
    restore_source_config_fields(args.model, args.output_dir)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("✅ Done!")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {args.output_dir}")
    print(f"  Quant time: {quant_time:.1f}s ({quant_time/3600:.2f}h)")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
