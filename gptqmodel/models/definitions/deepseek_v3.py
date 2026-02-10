# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..base import BaseQModel


class DeepSeekV3QModel(BaseQModel):
    # deepseek_v3 requires custom model code
    require_trust_remote_code = True

    require_fast_init = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # DeepSeek V3 uses dynamic modules based on lora(rank):
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_a_proj:0", "kv_a_proj_with_mqa:0", "q_b_proj:1", "kv_b_proj:1", "o_proj:2"),
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

    @classmethod
    def diceq_config_template(cls) -> Dict[str, Any]:
        from .glm4_moe_lite import Glm4MoeLiteQModel

        # 直接复用 glm4_moe_lite 的 DiceQ 默认配置，保证两者行为一致
        return Glm4MoeLiteQModel.diceq_config_template()

    def get_diceq_pipeline_config(self) -> Dict[str, Any]:
        cfg = self.diceq_config_template()
        override = self.quantize_config.extension_get("diceq")
        if isinstance(override, Dict):
            merged = copy.deepcopy(cfg)
            merged.update(override)
            cfg = merged
        layer_config = cfg.get("layer_config")
        if not isinstance(layer_config, dict) or not layer_config:
            cfg["layer_config"] = copy.deepcopy(self.diceq_config_template()["layer_config"])
            layer_config = cfg["layer_config"]

        global_subset_size = cfg.get("subset_size", None)
        if global_subset_size is not None:
            if isinstance(global_subset_size, bool):
                raise TypeError("diceq.subset_size must be int (>0) or float in (0,1].")
            for _, target_cfg in layer_config.items():
                if not isinstance(target_cfg, dict):
                    continue
                train_args = target_cfg.get("train_args")
                if not isinstance(train_args, dict):
                    train_args = {}
                    target_cfg["train_args"] = train_args
                if isinstance(global_subset_size, int):
                    if global_subset_size <= 0:
                        raise ValueError(f"diceq.subset_size(int) must be > 0, got {global_subset_size}")
                    train_args["subset_size"] = int(global_subset_size)
                    train_args.pop("train_subset_size", None)
                elif isinstance(global_subset_size, float):
                    ratio = float(global_subset_size)
                    if not (0.0 < ratio <= 1.0):
                        raise ValueError(f"diceq.subset_size(float) must be in (0,1], got {ratio}")
                    train_args["train_subset_size"] = ratio
                    train_args.pop("subset_size", None)
                else:
                    raise TypeError(
                        f"diceq.subset_size must be int or float, got {type(global_subset_size).__name__}"
                    )

        global_parallel_workers = cfg.get("parallel_workers", None)
        if global_parallel_workers is not None:
            global_parallel_workers = int(global_parallel_workers)
            if global_parallel_workers <= 0:
                raise ValueError(f"diceq.parallel_workers must be > 0, got {global_parallel_workers}")
            for _, target_cfg in layer_config.items():
                if not isinstance(target_cfg, dict):
                    continue
                train_args = target_cfg.get("train_args")
                if not isinstance(train_args, dict):
                    train_args = {}
                    target_cfg["train_args"] = train_args
                train_args["parallel_workers"] = global_parallel_workers

        if not cfg.get("targets"):
            cfg["targets"] = [k for k, v in layer_config.items() if isinstance(v, dict)]
        return cfg


class MLA(nn.Module):
    """Shared Multi-head Latent Attention fusion for DeepSeekV3 / glm4_moe_lite."""

    def __init__(self, config, ori_attention=None):
        super().__init__()
        self.config = config

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        if self.q_lora_rank is None:
            inp = hidden_states
        else:
            inp = self.q_a_layernorm(self.q_a_proj(hidden_states))

        q_pe = self.q_rope_proj(inp).view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim)
        q_content = self.q_absorbed_proj(inp).view(batch_size, seq_length, self.num_heads, self.kv_lora_rank)

        k_latent_raw = self.kv_a_proj_latent(hidden_states)
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_rot = self.kv_a_proj_rope(hidden_states)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_pe = q_pe.transpose(1, 2)
        if self.config.rope_interleave:
            q_pe, k_rot = self.apply_rotary_pos_emb_interleave(q_pe, k_rot, cos, sin)
        else:
            q_pe, k_rot = self.apply_rotary_pos_emb(q_pe, k_rot, cos, sin)
        q_pe = q_pe.transpose(1, 2)

        query_states = torch.cat((q_content, q_pe), dim=-1).transpose(1, 2) * self.scaling
        key_states = torch.cat((k_latent, k_rot.squeeze(1)), dim=-1).unsqueeze(1).transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_heads, dim=1)
        value_states = torch.repeat_interleave(k_latent.unsqueeze(1), self.num_heads, dim=1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states = key_states.transpose(2, 3)
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = key_states.transpose(2, 3)

        attn_weights = torch.matmul(query_states, key_states)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def _setup(self, cfg=None, ori_attention=None):
        if ori_attention.q_lora_rank is not None:
            ref_weight = ori_attention.q_a_proj.weight
        else:
            ref_weight = ori_attention.q_proj.weight
        device = ref_weight.device

        if ori_attention.q_lora_rank is None:
            self.q_proj = ori_attention.q_proj
        else:
            self.q_a_proj = ori_attention.q_a_proj
            self.q_a_layernorm = ori_attention.q_a_layernorm
            self.q_b_proj = ori_attention.q_b_proj

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

        with torch.no_grad():
            if device.type == "meta":
                raise RuntimeError(
                    "Cannot perform weight fusion on meta device. Please load the model with actual weights first."
                )

            if hasattr(self, "kv_a_proj_with_mqa"):
                W_KV_A = self.kv_a_proj_with_mqa.weight.to(device)
                has_bias = self.kv_a_proj_with_mqa.bias is not None

                split_idx = self.kv_lora_rank

                W_Latent = W_KV_A[:split_idx, :].contiguous()
                self.kv_a_proj_latent = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=has_bias)
                self.kv_a_proj_latent.weight = nn.Parameter(W_Latent)
                if has_bias:
                    self.kv_a_proj_latent.bias = nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[:split_idx].contiguous().to(device)
                    )

                W_Rope = W_KV_A[split_idx:, :].contiguous()
                self.kv_a_proj_rope = nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=has_bias)
                self.kv_a_proj_rope.weight = nn.Parameter(W_Rope)
                if has_bias:
                    self.kv_a_proj_rope.bias = nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[split_idx:].contiguous().to(device)
                    )

            W_Up = self.kv_b_proj.weight.to(device)
            D_latent = W_Up.shape[1]

            W_Up_view = W_Up.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, D_latent)
            W_UK = W_Up_view[:, :self.qk_nope_head_dim, :].clone()
            W_UV = W_Up_view[:, self.qk_nope_head_dim:, :].clone()

            W_Q_all = self.q_b_proj.weight.to(device)
            D_in = W_Q_all.shape[1]
            W_Q_view = W_Q_all.view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, D_in)
            W_Q_nope = W_Q_view[:, :self.qk_nope_head_dim, :]
            W_Q_rope = W_Q_view[:, self.qk_nope_head_dim:, :]

            W_Q_rope_flat = W_Q_rope.reshape(-1, D_in).contiguous()
            self.q_rope_proj = nn.Linear(D_in, self.num_heads * self.qk_rope_head_dim, bias=False)
            self.q_rope_proj.weight = nn.Parameter(W_Q_rope_flat.to(device))

            W_Q_absorbed = torch.einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            W_Q_absorbed_flat = W_Q_absorbed.reshape(-1, D_in).contiguous()
            self.q_absorbed_proj = nn.Linear(D_in, self.num_heads * self.kv_lora_rank, bias=False)
            self.q_absorbed_proj.weight = nn.Parameter(W_Q_absorbed_flat.to(device))

            W_O = self.o_proj.weight.to(device)
            W_O_view = W_O.view(self.hidden_size, self.num_heads, self.v_head_dim)
            W_Fused_VO = torch.einsum("xhd,hdc->xhc", W_O_view, W_UV)

            new_in_features = self.num_heads * self.kv_lora_rank
            W_Fused_Flat = W_Fused_VO.reshape(self.hidden_size, new_in_features).contiguous()

            has_bias = self.o_proj.bias is not None
            old_bias = self.o_proj.bias
            self.o_proj = nn.Linear(new_in_features, self.hidden_size, bias=has_bias)
            self.o_proj.weight = nn.Parameter(W_Fused_Flat.to(device))
            if has_bias:
                self.o_proj.bias = old_bias.to(device) if old_bias is not None else None

            del self.q_b_proj
            del self.kv_b_proj
            del self.kv_a_proj_with_mqa

        return self


class Glm4MoeLiteNaiveMoeNew(nn.Module):
    """Shared Naive MoE conversion used by DeepSeekV3 / glm4_moe_lite."""

    def __init__(self, config, ori_moe=None):
        super().__init__()
        self.config = config

        if ori_moe is not None:
            for attr in [
                "num_experts",
                "hidden_dim",
                "intermediate_dim",
            ]:
                if hasattr(ori_moe, attr):
                    setattr(self, attr, getattr(ori_moe, attr))
        else:
            self.num_experts = config.n_routed_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size

        self._setup(config, ori_moe)

    def _setup(self, cfg=None, ori_moe=None):
        if ori_moe is not None:
            dtype = ori_moe.down_proj.dtype
            device = ori_moe.down_proj.device
            self.act_fn = ori_moe.act_fn
        else:
            dtype = torch.float32
            device = "cpu"
            self.act_fn = nn.SiLU()

        self.gate_proj = nn.ModuleList([])
        self.up_proj = nn.ModuleList([])
        self.down_proj = nn.ModuleList([])

        with torch.no_grad():
            for i in range(self.num_experts):
                if ori_moe is not None:
                    gate_up = ori_moe.gate_up_proj[i]
                    mid_idx = gate_up.shape[0] // 2

                    gate_weight = gate_up[:mid_idx, :].contiguous()
                    up_weight = gate_up[mid_idx:, :].contiguous()
                    down_weight = ori_moe.down_proj[i].contiguous()
                else:
                    gate_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    up_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    down_weight = torch.empty(self.hidden_dim, self.intermediate_dim, dtype=dtype, device=device)

                self.gate_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False))
                self.gate_proj[i].weight = nn.Parameter(gate_weight.to(dtype=dtype, device=device))

                self.up_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False))
                self.up_proj[i].weight = nn.Parameter(up_weight.to(dtype=dtype, device=device))

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

