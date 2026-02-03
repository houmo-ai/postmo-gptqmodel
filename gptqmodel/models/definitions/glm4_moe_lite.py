# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
from typing import List, Optional

import torch
import torch.nn as nn

from ..base import BaseQModel
from ...utils.logger import setup_logger

log = setup_logger()


class Glm4MoeLiteQModel(BaseQModel):
    """
    GLM-4.7-Flash (glm4_moe_lite) Model with MLA (Multi-head Latent Attention) support.
    
    This model uses MLA architecture similar to DeepSeek-V2/MiniCPM3.
    Before quantization, we need to fuse the MLA weights for efficient inference:
    - Fuse kv_b_proj into q_absorbed_proj and o_proj
    - Split kv_a_proj_with_mqa into kv_a_proj_latent and kv_a_proj_rope
    
    Note: The MoE experts in this model use nn.Parameter (gate_up_proj, down_proj)
    instead of nn.Linear layers. The shared_experts use standard MLP structure.
    """
    
    # glm4_moe_lite requires custom model code
    require_trust_remote_code = True
    
    # allow dynamic expert index for layer_modules
    # config.n_routed_experts contains the actual expert count
    dynamic_expert_index = "n_routed_experts"
    
    pre_lm_head_norm_module = "model.norm"
    
    # Set to False since GLM-4.7-Flash may have dynamic module structures
    layer_modules_strict = False
    
    # MLA attention structure (after fusion):
    # - q_a_proj (if q_lora_rank is not None)
    # - q_rope_proj (fused from q_b_proj)
    # - q_absorbed_proj (fused from q_b_proj with kv_b_proj)
    # - kv_a_proj_latent (split from kv_a_proj_with_mqa)
    # - kv_a_proj_rope (split from kv_a_proj_with_mqa)
    # - o_proj (fused with kv_b_proj)
    #
    # MoE structure:
    # - shared_experts: standard MLP (gate_proj, up_proj, down_proj)
    # - experts: uses nn.Parameter (gate_up_proj, down_proj) - NOT quantizable as Linear
    # - gate: router, should not be quantized
    #
    # For layers with standard MLP (non-MoE), use gate_proj, up_proj, down_proj
    
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_proj:0",
                "q_rope_proj:1",
                "q_absorbed_proj:1",
                "kv_a_proj_latent:0",
                "kv_a_proj_rope:0",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                # MoE structure - shared_experts uses standard MLP
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Router should not be quantized
                "gate": ("gate:!",),
                # experts use nn.Parameter, not nn.Linear - mark as not quantizable
                "experts": ("experts:!",),
                # Standard MLP for dense layers (non-MoE)
                "gate_proj": ("gate_proj:0",),
                "up_proj": ("up_proj:0",),
                "down_proj": ("down_proj:1",),
            },
        }
    ]
    
    def pre_quantize_generate_hook_start(self):
        """
        Before quantization, fuse MLA weights for efficient inference.
        This converts the original MLA structure to a fused structure that:
        1. Reduces KV cache size by storing only latent vectors
        2. Absorbs kv_b_proj into q and o projections
        """
        super().pre_quantize_generate_hook_start()
        log.info("GLM4MoeLite: Starting MLA weight fusion for efficient inference...")
        self._fuse_mla_weights()
        log.info("GLM4MoeLite: MLA weight fusion completed.")
    
    def _fuse_mla_weights(self):
        """
        Fuse MLA (Multi-head Latent Attention) weights.
        
        The fusion process:
        1. Split kv_a_proj_with_mqa into kv_a_proj_latent and kv_a_proj_rope
        2. Extract W_UK and W_UV from kv_b_proj
        3. Fuse W_UK into q_b_proj to create q_absorbed_proj
        4. Create q_rope_proj from q_b_proj
        5. Fuse W_UV into o_proj
        6. Delete original kv_b_proj and q_b_proj
        """
        model = self.model
        
        # Get the layers module
        if hasattr(model, 'model'):
            layers = model.model.layers
        else:
            layers = model.layers
        
        fused_count = 0
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            
            # Skip if already fused
            if hasattr(attn, 'q_absorbed_proj'):
                continue
                
            # Skip if this layer doesn't have MLA structure (check for q_b_proj)
            if not hasattr(attn, 'q_b_proj'):
                # Check if it has q_proj instead (non-MLA attention)
                if hasattr(attn, 'q_proj'):
                    log.debug(f"GLM4MoeLite: Layer {layer_idx} uses standard attention, skipping MLA fusion.")
                continue
            
            # Must have kv_b_proj for MLA
            if not hasattr(attn, 'kv_b_proj'):
                log.warning(f"GLM4MoeLite: Layer {layer_idx} has q_b_proj but missing kv_b_proj, skipping.")
                continue
            
            self._fuse_layer_mla(attn, layer_idx)
            fused_count += 1
        
        log.info(f"GLM4MoeLite: Fused MLA weights for {fused_count} layers.")
    
    def _fuse_layer_mla(self, attn: nn.Module, layer_idx: int):
        """Fuse MLA weights for a single attention layer."""
        config = attn.config
        
        # Get dimensions
        num_heads = attn.num_heads
        kv_lora_rank = attn.kv_lora_rank
        qk_nope_head_dim = attn.qk_nope_head_dim
        qk_rope_head_dim = attn.qk_rope_head_dim
        v_head_dim = attn.v_head_dim
        hidden_size = config.hidden_size
        
        device = attn.kv_b_proj.weight.device
        dtype = attn.kv_b_proj.weight.dtype
        
        with torch.no_grad():
            # ========================================================================
            # Step 1: Split kv_a_proj_with_mqa into latent and rope projections
            # ========================================================================
            if hasattr(attn, "kv_a_proj_with_mqa"):
                W_KV_A = attn.kv_a_proj_with_mqa.weight  # [Latent + Rope, Hidden]
                has_bias = attn.kv_a_proj_with_mqa.bias is not None
                
                # Split point
                split_idx = kv_lora_rank
                
                # Create Latent projection layer
                W_Latent = W_KV_A[:split_idx, :].contiguous()
                attn.kv_a_proj_latent = nn.Linear(hidden_size, kv_lora_rank, bias=has_bias, device=device, dtype=dtype)
                attn.kv_a_proj_latent.weight = nn.Parameter(W_Latent)
                if has_bias:
                    attn.kv_a_proj_latent.bias = nn.Parameter(
                        attn.kv_a_proj_with_mqa.bias[:split_idx].contiguous()
                    )
                
                # Create RoPE projection layer
                W_Rope = W_KV_A[split_idx:, :].contiguous()
                attn.kv_a_proj_rope = nn.Linear(hidden_size, qk_rope_head_dim, bias=has_bias, device=device, dtype=dtype)
                attn.kv_a_proj_rope.weight = nn.Parameter(W_Rope)
                if has_bias:
                    attn.kv_a_proj_rope.bias = nn.Parameter(
                        attn.kv_a_proj_with_mqa.bias[split_idx:].contiguous()
                    )
            
            # ========================================================================
            # Step 2: Process kv_b_proj - Extract W_UK and W_UV
            # ========================================================================
            W_Up = attn.kv_b_proj.weight  # [NumHeads * (NopeDim + VDim), LatentDim]
            D_latent = W_Up.shape[1]
            
            # View as [NumHeads, NopeDim + VDim, LatentDim]
            W_Up_view = W_Up.view(num_heads, qk_nope_head_dim + v_head_dim, D_latent)
            
            # Extract W_UK (for Q fusion) -> [NumHeads, NopeDim, LatentDim]
            W_UK = W_Up_view[:, :qk_nope_head_dim, :].clone()
            
            # Extract W_UV (for O fusion) -> [NumHeads, VDim, LatentDim]
            W_UV = W_Up_view[:, qk_nope_head_dim:, :].clone()
            
            # ========================================================================
            # Step 3: Process q_b_proj - Create q_rope_proj and q_absorbed_proj
            # ========================================================================
            W_Q_all = attn.q_b_proj.weight  # [NumHeads * (NopeDim + RopeDim), Hidden]
            D_in = W_Q_all.shape[1]
            
            # View as [NumHeads, NopeDim + RopeDim, Hidden]
            W_Q_view = W_Q_all.view(num_heads, qk_nope_head_dim + qk_rope_head_dim, D_in)
            
            # Split Nope (Content) and Rope
            W_Q_nope = W_Q_view[:, :qk_nope_head_dim, :]  # [H, NopeDim, Hidden]
            W_Q_rope = W_Q_view[:, qk_nope_head_dim:, :]  # [H, RopeDim, Hidden]
            
            # Create RoPE-only Q projection
            W_Q_rope_flat = W_Q_rope.reshape(-1, D_in).contiguous()
            attn.q_rope_proj = nn.Linear(D_in, num_heads * qk_rope_head_dim, bias=False, device=device, dtype=dtype)
            attn.q_rope_proj.weight = nn.Parameter(W_Q_rope_flat)
            
            # Create Content (Absorbed) Q projection
            # Fusion: W_Fused = W_UK^T * W_Q_nope
            # Einsum: "hni, hnc -> hci" (h=Head, n=NopeDim, i=Hidden, c=Latent)
            W_Q_absorbed = torch.einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            
            # Flatten -> [H * LatentDim, Hidden]
            W_Q_absorbed_flat = W_Q_absorbed.reshape(-1, D_in).contiguous()
            attn.q_absorbed_proj = nn.Linear(D_in, num_heads * kv_lora_rank, bias=False, device=device, dtype=dtype)
            attn.q_absorbed_proj.weight = nn.Parameter(W_Q_absorbed_flat)
            
            # ========================================================================
            # Step 4: Process o_proj - Fuse with W_UV
            # ========================================================================
            W_O = attn.o_proj.weight  # [Hidden, NumHeads * VDim]
            
            # View as [Hidden, NumHeads, VDim]
            W_O_view = W_O.view(hidden_size, num_heads, v_head_dim)
            
            # Fusion: W_Fused = W_O * W_UV
            # Einsum: "xhd, hdc -> xhc" (x=Hidden, h=Head, d=VDim, c=Latent)
            W_Fused_VO = torch.einsum("xhd,hdc->xhc", W_O_view, W_UV)
            
            # Flatten -> [Hidden, NumHeads * LatentDim]
            new_in_features = num_heads * kv_lora_rank
            W_Fused_Flat = W_Fused_VO.reshape(hidden_size, new_in_features).contiguous()
            
            # Rebuild o_proj
            has_bias = attn.o_proj.bias is not None
            old_bias = attn.o_proj.bias
            attn.o_proj = nn.Linear(new_in_features, hidden_size, bias=has_bias, device=device, dtype=dtype)
            attn.o_proj.weight = nn.Parameter(W_Fused_Flat)
            if has_bias:
                attn.o_proj.bias = old_bias
            
            # ========================================================================
            # Step 5: Store scaling factor and clean up
            # ========================================================================
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            attn.kv_scale = 1 / math.sqrt(qk_head_dim)
            
            # Delete original layers
            del attn.q_b_proj
            del attn.kv_b_proj
            if hasattr(attn, 'kv_a_proj_with_mqa'):
                del attn.kv_a_proj_with_mqa
            
        log.debug(f"GLM4MoeLite: Layer {layer_idx} MLA fusion completed.")
