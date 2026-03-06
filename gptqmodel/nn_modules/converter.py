# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def _resolve_dtype_from_config(config):
    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if isinstance(dtype, torch.dtype):
        return dtype

    torch_dtype = getattr(config, "torch_dtype", None)
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    return None


class Qwen3_5MoeExpertsNew(nn.Module):
    """Qwen3.5-MoE experts rewritten as per-expert Linear modules for GPTQ."""

    def __init__(self, config, ori_experts=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        self.down_proj = nn.ModuleList()

        dtype = _resolve_dtype_from_config(config)
        linear_kwargs = {"bias": False}
        if ori_experts is not None:
            # Quantization path: avoid redundant allocation before assigning sliced weights.
            linear_kwargs["device"] = "meta"
        if dtype is not None:
            linear_kwargs["dtype"] = dtype

        for _ in range(self.num_experts):
            self.gate_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, **linear_kwargs))
            self.up_proj.append(nn.Linear(self.hidden_dim, self.intermediate_dim, **linear_kwargs))
            self.down_proj.append(nn.Linear(self.intermediate_dim, self.hidden_dim, **linear_kwargs))

        if ori_experts is not None:
            self._load_from_original(ori_experts)

    def _load_from_original(self, ori_experts) -> None:
        with torch.no_grad():
            for i in range(self.num_experts):
                gate_up = ori_experts.gate_up_proj[i].detach()
                down = ori_experts.down_proj[i].detach()

                gate_weight = gate_up[: self.intermediate_dim, :]
                up_weight = gate_up[self.intermediate_dim :, :]

                self.gate_proj[i].weight = nn.Parameter(gate_weight)
                self.up_proj[i].weight = nn.Parameter(up_weight)
                self.down_proj[i].weight = nn.Parameter(down)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            current_hidden_states = self.act_fn(self.gate_proj[expert_idx](current_state))
            current_hidden_states = current_hidden_states * self.up_proj[expert_idx](current_state)
            current_hidden_states = self.down_proj[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def convert_qwen3_5_moe_expert_converter(module, config):
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_moe_modeling

    to_replace = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, qwen3_5_moe_modeling.Qwen3_5MoeExperts):
            to_replace.append((name, sub_module))

    for name, sub_module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        new_module = Qwen3_5MoeExpertsNew(config=config, ori_experts=sub_module)
        setattr(parent, child_name, new_module)

    return module


def convert_gpt_oss_expert_converter(module, config):
    import torch.nn as nn
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling
    from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

    from ..models.definitions.gpt_oss import GptOssExpertsNew

    @use_kernel_forward_from_hub("MegaBlocksMoeMLP")
    class GptOssMLPNew(nn.Module):
        def __init__(self, config, ori_mlp=None):
            super().__init__()
            self.router = ori_mlp.router
            experts_new = GptOssExpertsNew(config, ori_mlp.experts)
            self.experts = experts_new

        def forward(self, hidden_states):
            router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
            routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
            return routed_out, router_scores

    # loop sub module to replace GptOssMLP with GptOssMLPNew
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, gpt_oss_modeling.GptOssMLP):
            new_module = GptOssMLPNew(config=config, ori_mlp=sub_module)
            setattr(module, name, new_module)

    return module

def convert_llama4_expert_converter(module, config):
    import torch
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP, Llama4TextMoe

    from ..utils.hf import no_init_weights

    # adapted/modified from https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/llama4.py
    class SequentialLlama4TextExperts(torch.nn.ModuleList):
        def __init__(self, config, original):
            self.num_experts = original.gate_up_proj.shape[0]
            with no_init_weights():
                super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])
            intermediate_size = original.down_proj.shape[1]

            with torch.inference_mode():
                # Batch process all expert parameters to avoid loops
                gate_up_batch = torch.stack([original.gate_up_proj[i] for i in range(self.num_experts)])
                down_batch = torch.stack([original.down_proj[i] for i in range(self.num_experts)])

                # Batch split and transpose
                gate_batch = gate_up_batch[:, :, :intermediate_size].transpose(-2, -1).contiguous()
                up_batch = gate_up_batch[:, :, intermediate_size:].transpose(-2, -1).contiguous()
                down_batch = down_batch.transpose(-2, -1).contiguous()

                # Batch assignment
                for i in range(self.num_experts):
                    self[i].gate_proj.weight.data = gate_batch[i]
                    self[i].up_proj.weight.data = up_batch[i]
                    self[i].down_proj.weight.data = down_batch[i]

    class SequentialLlama4TextMoe(torch.nn.Module):
        def __init__(self, config, original):
            super().__init__()
            self.top_k = config.num_experts_per_tok
            self.hidden_dim = config.hidden_size
            self.num_experts = config.num_local_experts
            self.experts = SequentialLlama4TextExperts(config, original.experts)
            self.router = original.router
            self.shared_expert = original.shared_expert

        def forward(self, hidden_states: torch.Tensor):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = self.router(hidden_states)
            if isinstance(router_logits, tuple):
                router_scores, router_logits = router_logits
                router_scores = router_scores.t()
            else:
                # transformers < 4.54.0 only returns router_logits
                router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

                router_scores = (
                    torch.full_like(router_logits, float("-inf"))
                    .scatter_(1, router_indices, router_top_value)
                    .transpose(0, 1)
                )
                router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

            out = self.shared_expert(hidden_states)
            for i in range(self.num_experts):
                out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

            return out, router_logits

    for name, sub_module in module.named_modules():
        if isinstance(sub_module, Llama4TextMoe):
            new_module = SequentialLlama4TextMoe(config=config.get_text_config(), original=sub_module)
            setattr(module, name, new_module)

    return module

def convert_glm4v_mlp_converter(module, config):
    import transformers.models.glm4v.modeling_glm4v as glm4v_modeling

    from ..models.definitions.glm4v import Glm4vTextMLPNew

    for name, sub_module in module.named_modules():
        if isinstance(sub_module, glm4v_modeling.Glm4vTextMLP):
            new_module = Glm4vTextMLPNew(config=config.get_text_config(), ori_mlp=sub_module)
            setattr(module, name, new_module)
    return module

MODULE_CONVERTER_MAP = {
    "llama4": convert_llama4_expert_converter,
    "gpt_oss": convert_gpt_oss_expert_converter,
    "qwen3_5_moe": convert_qwen3_5_moe_expert_converter,
    "qwen3_5_moe_text": convert_qwen3_5_moe_expert_converter,
    "glm4v": convert_glm4v_mlp_converter,
}
