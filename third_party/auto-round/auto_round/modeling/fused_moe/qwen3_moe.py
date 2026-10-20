# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # This must be linear for vllm alignment
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    @classmethod
    def from_original(cls, original, config):
        """Split HF fused experts while retaining the linear checkpoint layout.

        Like the Qwen3.5 fused adapter, gate/up are split along the output
        dimension. Do not release the original: callers may still use it.
        """
        with torch.device("meta"):
            replacement = cls(config)
        replacement.gate.weight = original.gate.weight
        for index, expert in enumerate(replacement.experts):
            gate, up = original.experts.gate_up_proj[index].chunk(2, dim=0)
            down = original.experts.down_proj[index]
            expert.gate_proj.weight = nn.Parameter(gate, requires_grad=original.experts.gate_up_proj.requires_grad)
            expert.up_proj.weight = nn.Parameter(up, requires_grad=original.experts.gate_up_proj.requires_grad)
            expert.down_proj.weight = nn.Parameter(down, requires_grad=original.experts.down_proj.requires_grad)
        replacement.train(original.training)
        return replacement

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        """Execute selected linear experts, as in the fused Qwen3.5 adapter."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        expert_output = self.experts_forward(hidden_states, selected_experts, routing_weights)
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)
