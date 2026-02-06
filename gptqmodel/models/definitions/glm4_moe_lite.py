# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor, device
import torch.nn.functional as F
import math
from ..base import BaseQModel

from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteAttention,Glm4MoeLiteNaiveMoe

@use_kernel_forward_from_hub("MegaGlm4MoeLiteAttention")
class MLA(nn.Module):
    def __init__(self, config, ori_attention: Glm4MoeLiteAttention | None = None):
        super().__init__()
        self.config = config

        # 从原始 attention 上拷贝必要属性
        for attr in [
            "use_cache",
            "layer_idx",
            "num_heads",
            "num_key_value_heads",
            "attention_dropout",
            "hidden_size",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "q_lora_rank",
            "scaling",
            "is_causal",
        ]:
            if hasattr(ori_attention, attr):
                setattr(self, attr, getattr(ori_attention, attr))

        self._setup(config, ori_attention)

    # ===== RoPE 辅助函数，直接复用你现有的实现 =====
    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_interleave(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        b, h, s, d = q.shape
        q = q.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    # ===== 核心前向：Q/K 等价，V+O 融合，支持 KV cache =====
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]]:
        batch_size, seq_length, hidden_dim = hidden_states.shape


        # =================================================================
        # 1. Query 生成：兵分两路 (RoPE vs Content)
        # =================================================================
        # Q-LoRA 的压缩层 (如果存在)
        if self.q_lora_rank is None:
            inp = hidden_states
        else:
            # 先过压缩层和 Norm
            inp = self.q_a_layernorm(self.q_a_proj(hidden_states))

        # 【路径 A】RoPE 部分 Query
        # q_pe: [B, S, H, D_rope]
        q_pe = self.q_rope_proj(inp).view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim)

        # 【路径 B】Content 部分 Query (已融合 W_UK)
        # q_content: [B, S, H, LatentDim]
        # 注意：这里直接输出了 Latent 维度的 Query，不需要再乘 W_UK
        q_content = self.q_absorbed_proj(inp).view(batch_size, seq_length, self.num_heads, self.kv_lora_rank)

        # =================================================================
        
        
        # 2. Key/Value 生成：Latent 空间 (压缩)
        # =================================================================
        # 生成压缩的 KV (Latent Vector)
        # A. 生成 Latent 向量 [B, S, LatentDim]
        k_latent_raw = self.kv_a_proj_latent(hidden_states)
        
        # 对 Latent 向量做 Layernorm (这是 MLA 的标准操作)
        k_latent = self.kv_a_layernorm(k_latent_raw)

        # B. 生成 RoPE 部分向量 [B, S, RopeDim]
        k_rot = self.kv_a_proj_rope(hidden_states)

        # =================================================================
        # 3. RoPE 位置编码
        # =================================================================
        # k_rot: [B, 1, S, RopeDim] (调整为 RoPE 接口需要的形状)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        # 为了 RoPE 计算，调整 q_pe 维度 -> [B, H, S, D]
        q_pe = q_pe.transpose(1, 2) 
        if self.config.rope_interleave:
            q_pe, k_rot = self.apply_rotary_pos_emb_interleave(q_pe, k_rot, cos, sin)
        else:
            q_pe, k_rot = self.apply_rotary_pos_emb(q_pe, k_rot, cos, sin)

        # q_pe 恢复形状 -> [B, S, H, D]
        q_pe = q_pe.transpose(1, 2)

        query_states = torch.cat((q_content, q_pe), dim=-1).transpose(1,2) * self.scaling
        key_states = torch.cat((k_latent, k_rot.squeeze(1)), dim=-1).unsqueeze(1).transpose(2, 3)
        key_states = torch.repeat_interleave(
            key_states,
            self.num_heads,
            dim=1,
        )
        value_states = torch.repeat_interleave(
            k_latent.unsqueeze(1),
            self.num_heads,
            dim=1,
        )
        # =================================================================
        # 4. KV Cache 管理 (极低显存占用)
        # =================================================================
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states = key_states.transpose(2, 3)  
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights: Optional[Tensor] =F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def _setup(self, cfg=None, ori_attention: Glm4MoeLiteAttention | None = None):
        # 取一个参考权重来确定 dtype/device
        if ori_attention.q_lora_rank is not None:
            ref_weight = ori_attention.q_a_proj.weight
        else:
            ref_weight = ori_attention.q_proj.weight
        dtype = ref_weight.dtype
        device = ref_weight.device

        # 引用原始 Q 子层
        if ori_attention.q_lora_rank is None:
            self.q_proj = ori_attention.q_proj
        else:
            self.q_a_proj = ori_attention.q_a_proj
            self.q_a_layernorm = ori_attention.q_a_layernorm
            self.q_b_proj = ori_attention.q_b_proj

        # 引用原始 K/V/A/O 子层
        self.kv_a_proj_with_mqa = ori_attention.kv_a_proj_with_mqa
        self.kv_a_layernorm = ori_attention.kv_a_layernorm
        self.kv_b_proj = ori_attention.kv_b_proj
        self.o_proj = ori_attention.o_proj

        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size


        # ===== V+O 权重融合：严格等价 =====
        with torch.no_grad():
            # 检查是否在 meta 设备上，如果是则无法进行权重融合
            if device.type == "meta":
                raise RuntimeError("Cannot perform weight fusion on meta device. Please load the model with actual weights first.")
            
            # ========================================================================
            # 第一步：拆分 kv_a_proj_with_mqa 
            # ========================================================================
            if hasattr(self, "kv_a_proj_with_mqa"):
                W_KV_A = self.kv_a_proj_with_mqa.weight.to(device) # [Latent + Rope, Hidden]
                has_bias = self.kv_a_proj_with_mqa.bias is not None
                
                # 确定切分点
                split_idx = self.kv_lora_rank
                
                # 1. 构造 Latent 投影层
                W_Latent = W_KV_A[:split_idx, :].contiguous()
                self.kv_a_proj_latent = torch.nn.Linear(self.hidden_size, self.kv_lora_rank, bias=has_bias)
                self.kv_a_proj_latent.weight = torch.nn.Parameter(W_Latent)
                if has_bias:
                    self.kv_a_proj_latent.bias = torch.nn.Parameter(self.kv_a_proj_with_mqa.bias[:split_idx].contiguous().to(device))
                # 2. 构造 RoPE 投影层
                W_Rope = W_KV_A[split_idx:, :].contiguous()
                self.kv_a_proj_rope = torch.nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=has_bias)
                self.kv_a_proj_rope.weight = torch.nn.Parameter(W_Rope)
                if has_bias:
                    self.kv_a_proj_rope.bias = torch.nn.Parameter(self.kv_a_proj_with_mqa.bias[split_idx:].contiguous().to(device))

            # ========================================================================
            # 第二步：处理 kv_b_proj (必须最先做，因为 Q 融合依赖于 W_UK)
            # ========================================================================
            # 原始权重形状: [NumHeads * (NopeDim + VDim), LatentDim]
            # 注意：PyTorch Linear 权重是 [Out, In]
            W_Up = self.kv_b_proj.weight.to(device) 
            D_latent = W_Up.shape[1]

            # View 成 [NumHeads, NopeDim + VDim, LatentDim]
            W_Up_view = W_Up.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, D_latent)
            
            # 提取 W_UK (用于 Q 融合) -> [NumHeads, NopeDim, LatentDim]
            W_UK = W_Up_view[:, :self.qk_nope_head_dim, :].clone()
            
            # 提取 W_UV (用于 O 融合) -> [NumHeads, VDim, LatentDim]
            W_UV = W_Up_view[:, self.qk_nope_head_dim:, :].clone()

            # ========================================================================
            # 第三步：处理 q_b_proj (依赖 W_UK)
            # ========================================================================
            W_Q_all = self.q_b_proj.weight.to(device) # [NumHeads * (NopeDim + RopeDim), Hidden]
            D_in = W_Q_all.shape[1]
            
            # View 成 [NumHeads, NopeDim + RopeDim, Hidden]
            W_Q_view = W_Q_all.view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, D_in)
            
            # 拆分 Nope (Content) 和 Rope
            W_Q_nope = W_Q_view[:, :self.qk_nope_head_dim, :] # [H, NopeDim, Hidden]
            W_Q_rope = W_Q_view[:, self.qk_nope_head_dim:, :] # [H, RopeDim, Hidden]
            
            # --- 2.1 构造 RoPE 专用层 ---
            # 直接展平 W_Q_rope -> [H * RopeDim, Hidden]
            W_Q_rope_flat = W_Q_rope.reshape(-1, D_in).contiguous()
            self.q_rope_proj = torch.nn.Linear(D_in, self.num_heads * self.qk_rope_head_dim, bias=False)
            self.q_rope_proj.weight = torch.nn.Parameter(W_Q_rope_flat.to(device))
            
            # --- 2.2 构造 Content (Absorbed) 专用层 ---
            # 融合公式: W_Fused = W_UK^T * W_Q_nope
            # 维度: [H, Latent, Hidden] = [H, Nope, Hidden] * [H, Nope, Latent] (转置后)
            # Einsum: "hni, hnc -> hci" (h=Head, n=NopeDim, i=Hidden, c=Latent)
            W_Q_absorbed = torch.einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            
            # 展平 -> [H * LatentDim, Hidden]
            W_Q_absorbed_flat = W_Q_absorbed.reshape(-1, D_in).contiguous()
            self.q_absorbed_proj = torch.nn.Linear(D_in, self.num_heads * self.kv_lora_rank, bias=False)
            self.q_absorbed_proj.weight = torch.nn.Parameter(W_Q_absorbed_flat.to(device))

            # ========================================================================
            # 第四步：处理 o_proj (依赖 W_UV)
            # ========================================================================
            W_O = self.o_proj.weight.to(device) # [Hidden, NumHeads * VDim]
            
            # View 成 [Hidden, NumHeads, VDim]
            W_O_view = W_O.view(self.hidden_size, self.num_heads, self.v_head_dim)
            
            # 融合公式: W_Fused = W_O * W_UV
            # Einsum: "xhd, hdc -> xhc" (x=Hidden, h=Head, d=VDim, c=Latent)
            W_Fused_VO = torch.einsum("xhd,hdc->xhc", W_O_view, W_UV)
            
            # 展平 -> [Hidden, NumHeads * LatentDim]
            new_in_features = self.num_heads * self.kv_lora_rank
            W_Fused_Flat = W_Fused_VO.reshape(self.hidden_size, new_in_features).contiguous()
            
            # 重建 out  Linear 层
            has_bias = self.o_proj.bias is not None
            old_bias = self.o_proj.bias
            self.o_proj = torch.nn.Linear(new_in_features, self.hidden_size, bias=has_bias)
            self.o_proj.weight = torch.nn.Parameter(W_Fused_Flat.to(device))
            if has_bias:
                self.o_proj.bias = old_bias.to(device) if old_bias is not None else None

            # ========================================================================
            # 第五步：清理旧层
            # ========================================================================
            del self.q_b_proj
            del self.kv_b_proj
            del self.kv_a_proj_with_mqa

        return self


@use_kernel_forward_from_hub("MegaGlm4MoeLiteNaiveMoe")
class Glm4MoeLiteNaiveMoeNew(nn.Module):
    def __init__(self, config, ori_moe: Glm4MoeLiteNaiveMoe | None = None):
        super().__init__()
        self.config = config

        # 从原始 attention 上拷贝必要属性，或从 config 中获取
        if ori_moe is not None:
            for attr in [
                "num_experts",
                "hidden_dim",
                "intermediate_dim",
            ]:
                if hasattr(ori_moe, attr):
                    setattr(self, attr, getattr(ori_moe, attr))
        else:
            # 从配置中获取属性（加载量化模型时）
            self.num_experts = config.n_routed_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size

        self._setup(config, ori_moe)

    def _setup(self, cfg=None, ori_moe: Glm4MoeLiteNaiveMoe | None = None):
        if ori_moe is not None:
            # 从原始 MoE 转换（量化时）
            dtype = ori_moe.down_proj.dtype
            device = ori_moe.down_proj.device
            self.act_fn = ori_moe.act_fn
        else:
            # 从配置直接创建（加载量化模型时）
            # 权重会在后续加载量化权重时被填充
            dtype = torch.float32
            device = "cpu"
            self.act_fn = nn.SiLU()
        
        self.gate_proj = nn.ModuleList([])
        self.up_proj = nn.ModuleList([])
        self.down_proj = nn.ModuleList([])
        
        with torch.no_grad():
            for i in range(self.num_experts):
                if ori_moe is not None:
                    # 从原始 MoE 转换
                    gate_up = ori_moe.gate_up_proj[i]  # [2*I, H]
                    mid_idx = gate_up.shape[0] // 2
                    
                    # gate_up_proj 拆分 (不转置)
                    gate_weight = gate_up[:mid_idx, :].contiguous()  # [I, H]
                    up_weight = gate_up[mid_idx:, :].contiguous()    # [I, H]
                    
                    # down_proj 也不转置! 原始就是 [H, I]
                    down_weight = ori_moe.down_proj[i].contiguous()  # [H, I] ✅ 保持原样
                else:
                    # 创建空的 Linear 层，权重会在加载量化权重时被填充
                    gate_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    up_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    down_weight = torch.empty(self.hidden_dim, self.intermediate_dim, dtype=dtype, device=device)
                
                # 创建 Linear 层
                self.gate_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False))
                self.gate_proj[i].weight = nn.Parameter(gate_weight.to(dtype=dtype, device=device))
                
                self.up_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False))
                self.up_proj[i].weight = nn.Parameter(up_weight.to(dtype=dtype, device=device))
                
                # down_proj: Linear(intermediate_dim -> hidden_dim)
                # 权重格式: [hidden_dim, intermediate_dim] = [H, I]
                self.down_proj.append(nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False))
                self.down_proj[i].weight = nn.Parameter(down_weight.to(dtype=dtype, device=device))
        
        return self
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def convert_glm4_moe_lite_convert(module, config):
    # 遍历模块树，替换 HF 原始注意力层
    # to_replace = []
    # for name, sub_module in module.named_modules():
    #     if isinstance(sub_module, Glm4MoeLiteAttention):
    #         to_replace.append((name, sub_module))

    # for name, sub_module in to_replace:
    #     parent_name, _, child_name = name.rpartition(".")
    #     parent = module.get_submodule(parent_name) if parent_name else module
    #     new_module = MLA(config=config, ori_attention=sub_module)
    #     setattr(parent, child_name, new_module)
    # torch.cuda.empty_cache()

    # 遍历模块树，替换 HF 原始注意力层
    to_replace = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, Glm4MoeLiteNaiveMoe):
            to_replace.append((name, sub_module))

    for name, sub_module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        new_module = Glm4MoeLiteNaiveMoeNew(config=config, ori_moe=sub_module)
        setattr(parent, child_name, new_module)
    torch.cuda.empty_cache()
    return module


class Glm4MoeLiteQModel(BaseQModel):    
    # glm4_moe_lite requires custom model code
    require_trust_remote_code = True
    
    # allow dynamic expert index for layer_modules
    # config.n_routed_experts contains the actual expert count
    dynamic_expert_index = "n_routed_experts"  # 修复：使用 config 中的正确属性名
    
    pre_lm_head_norm_module = "model.norm"
    support_offload_to_disk = False
    
    # Set to False since GLM-4.7-Flash may have dynamic module structures
    layer_modules_strict = False
    
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # 原始HF结构的模块名（因为attention转换被注释掉了）
                "q_a_proj:0",
                "q_b_proj:1",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:?": {
                # MoE structure - shared_experts uses standard MLP
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Router should not be quantized (it's a nn.Linear with weight as router)
                "gate": ("gate:!",),
                # experts 转换后结构是: experts.gate_proj[i], experts.up_proj[i], experts.down_proj[i]
                # 路径直接是 experts.gate_proj.0, experts.gate_proj.1 等
                # 使用 {expert_index} 占位符，由 build_moe_modules_if_need 展开
                "experts": (
                    "gate_proj.{expert_index}:0",
                    "up_proj.{expert_index}:0",
                    "down_proj.{expert_index}:1",
                ),
                # Standard MLP for dense layers (non-MoE)
                "gate_proj": ("gate_proj:0",),
                "up_proj": ("up_proj:0",),
                "down_proj": ("down_proj:1",),
            },
        }
    ]
    
    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm4_moe_lite_modeling

            glm4_moe_lite_modeling.Glm4MoeLiteNaiveMoe = Glm4MoeLiteNaiveMoeNew

    def pre_quantize_generate_hook_start(self):
        """
        Before quantization, fuse MLA weights for efficient inference.
        This converts the original MLA structure to a fused structure that:
        1. Reduces KV cache size by storing only latent vectors
        2. Absorbs kv_b_proj into q and o projections
        """
        super().pre_quantize_generate_hook_start()

        pass

    def after_model_load(self, model, load_quantized_model):
        self.model = super().after_model_load(model, load_quantized_model)
        
        # 只在首次加载未量化模型时执行MoE结构转换
        # 加载量化模型时，由于 before_model_load 已经进行了 monkey patch，
        # from_config 创建的模型已经是 Glm4MoeLiteNaiveMoeNew 结构
        if not load_quantized_model:
            self.model = convert_glm4_moe_lite_convert(self.model, self.config)
        
        return self.model
    
    
