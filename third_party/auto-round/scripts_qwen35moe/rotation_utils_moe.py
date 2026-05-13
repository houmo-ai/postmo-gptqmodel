"""Hadamard rotation utilities for Qwen3.5-MoE (35B-A3B) models.

Provides offline Hadamard rotation for MoE LLM backbone and optional
vision module rotation. Adapted from:
  - scripts/quantize.py (Qwen3.5 dense rotation)
  - gptqmodel rotation.py (Qwen3.5-MoE rotation functions)

Key differences from dense Qwen3.5:
  - MLP is MoE: gate (router), shared_expert, per-expert gate/up/down_proj
  - RMSNorm uses (1 + weight) scaling (Qwen3_5MoeRMSNorm)
  - 256 experts with 8 active per token
  - Hybrid attention: linear_attn + self_attn every 4th layer

Usage:
    from scripts_qwen35moe.rotation_utils_moe import (
        apply_hadamard_rotation_moe,
        fuse_llm_norms_moe,
        rotate_llm_moe,
    )
"""

import sys
import torch


# ─── Hadamard matrix generation ──────────────────────────────────────────────

def _get_hadamard_matrix(dim, device):
    """Generate normalized Hadamard matrix of given dimension.

    For power-of-2 dimensions, builds exact Hadamard via Sylvester construction.
    Otherwise falls back to QR decomposition of random Gaussian matrix.
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


# ─── Norm helpers (Qwen3_5MoeRMSNorm uses 1+weight) ─────────────────────────

_ADDITIVE_NORM_CLASSES = {
    "Qwen3_5MoeRMSNorm",
    "Qwen3_5RMSNorm",
    "Qwen3NextRMSNorm",
}


def _get_norm_scale(layernorm):
    """Get multiplicative scale from LayerNorm/RMSNorm.
    Qwen3.5-MoE RMSNorm uses (1 + weight) instead of just weight.
    """
    weight = layernorm.weight.data.double()
    if layernorm.__class__.__name__ in _ADDITIVE_NORM_CLASSES:
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
    if layernorm.__class__.__name__ in _ADDITIVE_NORM_CLASSES:
        layernorm.weight.data.zero_()
    else:
        layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, 'bias') and layernorm.bias is not None:
        layernorm.bias.data.zero_()


# ─── Weight rotation helpers ─────────────────────────────────────────────────

def _rotate_weight(W, Q, device):
    """Apply rotation: W_new = W @ Q"""
    dtype = W.dtype
    return (W.to(device=device, dtype=torch.float64) @ Q).to(dtype=dtype).contiguous()


def _rotate_weight_out(W, Q, device):
    """Apply rotation: W_new = Q^T @ W"""
    dtype = W.dtype
    return (Q.T @ W.to(device=device, dtype=torch.float64)).to(dtype=dtype).contiguous()


def _bake_mean_into_linear(linear):
    """Subtract mean from linear weight columns (LayerNorm mean removal)."""
    dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(dtype)


def _bake_mean_into_conv(conv):
    """Subtract per-output-channel mean from conv weight."""
    dtype = conv.weight.dtype
    W = conv.weight.data.double()
    W_flat = W.reshape(W.shape[0], -1)
    W_flat = W_flat - W_flat.mean(dim=-1, keepdim=True)
    conv.weight.data = W_flat.reshape(W.shape).to(dtype)


# ─── Packed tensor helpers for MoE experts ──────────────────────────────────

def _fuse_norm_into_packed_tensor(layernorm, tensor_param):
    """Fuse norm scale into packed tensor's last dimension.

    For Qwen3_5MoeExperts, gate_up_proj has shape (num_experts, 2*inter, hidden)
    and the norm scale has shape (hidden,).
    """
    scale = _get_norm_scale(layernorm)
    dtype = tensor_param.dtype
    tensor_param.data = (tensor_param.data.double() * scale).to(dtype)


def _bake_mean_into_packed_down_proj(down_proj_param):
    """Subtract per-expert mean from packed down_proj.

    down_proj shape: (num_experts, hidden_dim, intermediate_dim)
    For each expert, subtract mean across intermediate_dim (last dim).
    """
    dtype = down_proj_param.dtype
    W = down_proj_param.data.double()
    down_proj_param.data = (W - W.mean(dim=-1, keepdim=True)).to(dtype)


# ─── Model structure helpers ─────────────────────────────────────────────────

def _get_visual(model):
    """Get visual module from Qwen3.5-MoE VLM model."""
    if hasattr(model, 'visual'):
        return model.visual
    if hasattr(model, 'model'):
        if hasattr(model.model, 'visual'):
            return model.model.visual
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'visual'):
            return model.model.model.visual
    raise AttributeError(f"Cannot find visual module in {type(model).__name__}")


def _get_llm_parts(model):
    """Get LLM layers, embed_tokens, lm_head, final_norm from model.

    Supports:
      - VLM (Qwen3_5MoeForConditionalGeneration):
            model.model.language_model.{layers, embed_tokens, norm} + model.lm_head
      - CausalLM (Qwen3_5MoeForCausalLM):
            model.model.{layers, embed_tokens, norm} + model.lm_head
    """
    lm_head = getattr(model, 'lm_head', None)

    # VLM path: model.model.language_model
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        if hasattr(lm, 'layers'):
            return lm.layers, lm.embed_tokens, lm_head, lm.norm

    # CausalLM path: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        m = model.model
        return m.layers, m.embed_tokens, lm_head, m.norm

    raise ValueError(f"Cannot find LLM layers in {type(model).__name__}")


# ─── LLM Norm Fusion (MoE) ──────────────────────────────────────────────────

def fuse_llm_norms_moe(model, device):
    """Fuse all LLM RMSNorm weights into adjacent linear layers (MoE version).

    Handles:
      - final_norm → lm_head
      - post_attention_layernorm → gate (router), shared_expert, per-expert (packed tensor)
      - input_layernorm → attention projections (self_attn or linear_attn)

    Experts may use packed format: gate_up_proj (num_experts, 2*inter, hidden),
    or per-expert modules with gate_proj/up_proj.
    """
    layers, embed, lm_head, final_norm = _get_llm_parts(model)

    # Fuse final norm → lm_head
    _fuse_norm_into_linears(final_norm, [lm_head])
    _reset_norm(final_norm)

    for layer in layers:
        post_ln = layer.post_attention_layernorm
        input_ln = layer.input_layernorm
        mlp = layer.mlp

        # Fuse post_attention_layernorm → MoE components
        # 1. Router gate
        if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            dtype = mlp.gate.weight.dtype
            W_ = mlp.gate.weight.data.double()
            scale = _get_norm_scale(post_ln)
            mlp.gate.weight.data = (W_ * scale).to(dtype)

        # 2. Shared expert
        if hasattr(mlp, "shared_expert"):
            _fuse_norm_into_linears(post_ln, [mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj])

        # 3. Shared expert gate (scalar or linear)
        if hasattr(mlp, "shared_expert_gate") and hasattr(mlp.shared_expert_gate, "weight"):
            _fuse_norm_into_linears(post_ln, [mlp.shared_expert_gate])

        # 4. Per-expert: handle packed or per-module format
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            if hasattr(experts, "gate_up_proj") and isinstance(experts.gate_up_proj, torch.nn.Parameter):
                # Packed format: gate_up_proj shape (num_experts, 2*inter, hidden)
                _fuse_norm_into_packed_tensor(post_ln, experts.gate_up_proj)
            elif hasattr(experts, "gate_proj") and hasattr(experts, "up_proj"):
                for i in range(len(experts.gate_proj)):
                    _fuse_norm_into_linears(post_ln, [experts.gate_proj[i], experts.up_proj[i]])
            else:
                num_experts = getattr(experts, "num_experts", None)
                if num_experts is not None:
                    for i in range(num_experts):
                        expert = experts[i]
                        linears = []
                        if hasattr(expert, "gate_proj"):
                            linears.append(expert.gate_proj)
                        if hasattr(expert, "up_proj"):
                            linears.append(expert.up_proj)
                        if linears:
                            _fuse_norm_into_linears(post_ln, linears)

        # Fuse input_layernorm → attention projections
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


# ─── LLM Rotation (MoE) ─────────────────────────────────────────────────────

def rotate_llm_moe(model, Q, device):
    """Apply Hadamard rotation to MoE LLM layers.

    Rotates:
      - embed_tokens, lm_head
      - Attention: q/k/v_proj (or linear_attn projections) input side
      - Attention: o_proj (or out_proj) output side
      - MoE: gate (router), shared_expert gate/up/down_proj
      - MoE packed experts: gate_up_proj (matmul on last dim), down_proj (Q^T on dim 1)
    """
    layers, embed, lm_head, _ = _get_llm_parts(model)

    # Rotate embeddings and head
    embed.weight.data = _rotate_weight(embed.weight.data, Q, device)
    lm_head.weight.data = _rotate_weight(lm_head.weight.data, Q, device)

    for i, layer in enumerate(layers):
        print(f"  Rotating MoE LLM layer {i}/{len(layers)}", end="\r")
        mlp = layer.mlp

        # ── Attention input rotation (W_new = W @ Q) ──
        if hasattr(layer, "self_attn"):
            for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
                proj.weight.data = _rotate_weight(proj.weight.data, Q, device)

        if hasattr(layer, "linear_attn"):
            for proj in [layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_z,
                         layer.linear_attn.in_proj_b, layer.linear_attn.in_proj_a]:
                proj.weight.data = _rotate_weight(proj.weight.data, Q, device)

        # ── Attention output rotation (W_new = Q^T @ W) ──
        if hasattr(layer, "self_attn"):
            _bake_mean_into_linear(layer.self_attn.o_proj)
            layer.self_attn.o_proj.weight.data = _rotate_weight_out(
                layer.self_attn.o_proj.weight.data, Q, device)

        if hasattr(layer, "linear_attn"):
            _bake_mean_into_linear(layer.linear_attn.out_proj)
            layer.linear_attn.out_proj.weight.data = _rotate_weight_out(
                layer.linear_attn.out_proj.weight.data, Q, device)

        # ── MoE MLP input rotation (W_new = W @ Q) ──
        # Router gate
        if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            mlp.gate.weight.data = _rotate_weight(mlp.gate.weight.data, Q, device)

        # Shared expert input
        if hasattr(mlp, "shared_expert"):
            mlp.shared_expert.gate_proj.weight.data = _rotate_weight(
                mlp.shared_expert.gate_proj.weight.data, Q, device)
            mlp.shared_expert.up_proj.weight.data = _rotate_weight(
                mlp.shared_expert.up_proj.weight.data, Q, device)

        # Shared expert gate
        if hasattr(mlp, "shared_expert_gate") and hasattr(mlp.shared_expert_gate, "weight"):
            mlp.shared_expert_gate.weight.data = _rotate_weight(
                mlp.shared_expert_gate.weight.data, Q, device)

        # Per-expert input: handle packed or per-module format
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            if hasattr(experts, "gate_up_proj") and isinstance(experts.gate_up_proj, torch.nn.Parameter):
                # Packed format: gate_up_proj shape (num_experts, 2*inter, hidden)
                dtype = experts.gate_up_proj.dtype
                gate_up = experts.gate_up_proj.data.to(device=device, dtype=torch.float64)
                experts.gate_up_proj.data = torch.matmul(gate_up, Q).to(dtype=dtype)
            elif hasattr(experts, "gate_proj") and hasattr(experts, "up_proj"):
                for ei in range(len(experts.gate_proj)):
                    experts.gate_proj[ei].weight.data = _rotate_weight(
                        experts.gate_proj[ei].weight.data, Q, device)
                    experts.up_proj[ei].weight.data = _rotate_weight(
                        experts.up_proj[ei].weight.data, Q, device)
            else:
                num_experts = getattr(experts, "num_experts", None)
                if num_experts is not None:
                    for ei in range(num_experts):
                        expert = experts[ei]
                        if hasattr(expert, "gate_proj"):
                            expert.gate_proj.weight.data = _rotate_weight(
                                expert.gate_proj.weight.data, Q, device)
                        if hasattr(expert, "up_proj"):
                            expert.up_proj.weight.data = _rotate_weight(
                                expert.up_proj.weight.data, Q, device)

        # ── MoE MLP output rotation (W_new = Q^T @ W) ──
        # Shared expert output
        if hasattr(mlp, "shared_expert"):
            _bake_mean_into_linear(mlp.shared_expert.down_proj)
            mlp.shared_expert.down_proj.weight.data = _rotate_weight_out(
                mlp.shared_expert.down_proj.weight.data, Q, device)

        # Per-expert output: handle packed or per-module format
        if experts is not None:
            if hasattr(experts, "down_proj") and isinstance(experts.down_proj, torch.nn.Parameter):
                # Packed format: down_proj shape (num_experts, hidden, inter)
                _bake_mean_into_packed_down_proj(experts.down_proj)
                dtype = experts.down_proj.dtype
                down = experts.down_proj.data.to(device=device, dtype=torch.float64)
                experts.down_proj.data = torch.matmul(Q.T, down).to(dtype=dtype)
            elif hasattr(experts, "down_proj") and not isinstance(experts.down_proj, torch.nn.Parameter):
                for ei in range(len(experts.down_proj)):
                    _bake_mean_into_linear(experts.down_proj[ei])
                    experts.down_proj[ei].weight.data = _rotate_weight_out(
                        experts.down_proj[ei].weight.data, Q, device)
            else:
                num_experts = getattr(experts, "num_experts", None)
                if num_experts is not None:
                    for ei in range(num_experts):
                        expert = experts[ei]
                        if hasattr(expert, "down_proj"):
                            _bake_mean_into_linear(expert.down_proj)
                            expert.down_proj.weight.data = _rotate_weight_out(
                                expert.down_proj.weight.data, Q, device)

    print(f"  Rotated {len(layers)} MoE LLM layers")


# ─── Vision Norm Fusion ──────────────────────────────────────────────────────

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

    merger_norm = visual.merger.norm
    merger_fc1 = visual.merger.linear_fc1
    _fuse_ln_linear(merger_norm, [merger_fc1], tile_factor=merger_fc1.in_features // merger_norm.weight.shape[0])
    _reset_norm(merger_norm)


# ─── Vision Rotation ─────────────────────────────────────────────────────────

def rotate_vision_full(model, Q_vision, device):
    """Rotate full vision encoder (patch_embed + blocks + merger input)."""
    visual = _get_visual(model)

    # Rotate patch embed conv output → rotated hidden space
    dtype = visual.patch_embed.proj.weight.dtype
    W = visual.patch_embed.proj.weight.data.double().to(device)
    W_flat = W.reshape(W.shape[0], -1)
    W_flat = Q_vision.T @ W_flat
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
        _bake_mean_into_linear(block.attn.proj)
        block.attn.proj.weight.data = _rotate_weight_out(block.attn.proj.weight.data, Q_vision, device)
        block.mlp.linear_fc1.weight.data = _rotate_weight(block.mlp.linear_fc1.weight.data, Q_vision, device)
        _bake_mean_into_linear(block.mlp.linear_fc2)
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
    _bake_mean_into_linear(visual.merger.linear_fc2)
    visual.merger.linear_fc2.weight.data = _rotate_weight_out(
        visual.merger.linear_fc2.weight.data, Q_llm, device)


# ─── Vision layer names ─────────────────────────────────────────────────────

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


# ─── High-level API ──────────────────────────────────────────────────────────

def apply_hadamard_rotation_moe(model, rotate_vision="merger_only", device="cuda:0", seed=42):
    """Apply Hadamard rotation to Qwen3.5-MoE model.

    Args:
        model: Qwen3_5MoeForConditionalGeneration or CausalLM model.
        rotate_vision: "none", "merger_only", or "full".
        device: Device for rotation computation.
        seed: Random seed for reproducible Q matrices.

    Returns:
        (model, Q_llm, Q_vision) where Q_vision is None if no vision rotation.
    """
    dev = torch.device(device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    text_config = getattr(model.config, 'text_config', model.config)
    hidden_size = text_config.hidden_size
    Q_llm = _get_hadamard_matrix(hidden_size, dev)
    print(f"[Rotation] Q_llm shape: {Q_llm.shape} (hidden_size={hidden_size})")

    has_vision = False
    try:
        _get_visual(model)
        has_vision = True
    except AttributeError:
        pass

    # Step 1: Fuse LLM norms
    print("[Rotation] Step 1: Fusing LLM norms (MoE)...")
    fuse_llm_norms_moe(model, dev)

    Q_vision = None

    # Step 2: Vision rotation
    if rotate_vision != "none" and has_vision:
        if rotate_vision == "full":
            print("[Rotation] Step 2: Fusing vision norms + full encoder rotation...")
            fuse_vision_norms(model, dev)

            vis_config = getattr(model.config, 'vision_config', None)
            vis_hidden = vis_config.hidden_size if vis_config else _get_visual(model).blocks[0].attn.qkv.in_features
            Q_vision = _get_hadamard_matrix(vis_hidden, dev)
            print(f"  Q_vision shape: {Q_vision.shape} (vis_hidden={vis_hidden})")

            rotate_vision_full(model, Q_vision, device=dev)

        print(f"[Rotation] Rotating merger output → LLM space...")
        rotate_merger_output(model, Q_llm, device=dev)
    elif rotate_vision != "none":
        print("[Rotation] No vision module detected, skipping vision rotation.")

    # Step 3: Rotate LLM
    print("[Rotation] Step 3: Rotating MoE LLM layers...")
    rotate_llm_moe(model, Q_llm, dev)

    print("[Rotation] Done.")
    return model, Q_llm, Q_vision
