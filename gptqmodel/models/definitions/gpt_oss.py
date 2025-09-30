# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class GptOssGPTQ(BaseGPTQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_local_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "GptOssDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],

        ["mlp.router.proj"],

        # uses dynamic_expert_index
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_up_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],
    ]

    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": {
                "router": ("proj",),
                "experts": {
                    "#": ("gate_up_proj", "down_proj"),
                },
            },
        }
    ]

    require_monkeypatch = True

    def monkey_patch(self):
        import torch
        import torch.nn as nn

        # Convert router (Parameter -> nn.Linear named `proj`)
        for layer in getattr(self.model, "layers", []):
            router = getattr(layer.mlp, "router", None)
            if router is not None and not hasattr(router, "proj"):
                # router has weight: (num_experts, hidden), bias: (num_experts)
                weight = getattr(router, "weight", None)
                bias = getattr(router, "bias", None)
                if isinstance(weight, torch.nn.Parameter) and isinstance(bias, torch.nn.Parameter):
                    num_experts, hidden = weight.shape
                    proj = nn.Linear(hidden, num_experts, bias=True)
                    with torch.no_grad():
                        proj.weight.copy_(weight)
                        proj.bias.copy_(bias)
                    router.proj = proj

                    def router_forward(self, hidden_states):
                        return torch.nn.functional.linear(hidden_states, self.proj.weight, self.proj.bias).softmax(dim=-1), None

                    # keep original behavior: return (scores, indices) for compatibility with upstream
                    def router_forward_topk(self, hidden_states):
                        hidden_states = hidden_states.reshape(-1, self.proj.in_features)
                        logits = torch.nn.functional.linear(hidden_states, self.proj.weight, self.proj.bias)
                        topk = getattr(self, "top_k", 2)
                        values, indices = torch.topk(logits, topk, dim=-1)
                        values = torch.nn.functional.softmax(values, dim=1, dtype=values.dtype)
                        scores = torch.zeros_like(logits).scatter_(1, indices, values)
                        return scores, indices

                    # prefer full-compat forward signature
                    router.forward = router_forward_topk.__get__(router, type(router))

        # Convert experts tensors (batched Parameters -> per-expert nn.Linear)
        for layer in getattr(self.model, "layers", []):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue

            # Expected attributes on original experts:
            gate_up = getattr(experts, "gate_up_proj", None)
            gate_up_bias = getattr(experts, "gate_up_proj_bias", None)
            down = getattr(experts, "down_proj", None)
            down_bias = getattr(experts, "down_proj_bias", None)

            # If already converted, skip
            if isinstance(getattr(experts, "experts", None), nn.ModuleList):
                continue

            # Validate shapes
            if not (isinstance(gate_up, nn.Parameter) and isinstance(gate_up_bias, nn.Parameter)
                    and isinstance(down, nn.Parameter) and isinstance(down_bias, nn.Parameter)):
                continue

            num_experts, hidden, twoE = gate_up.shape
            expert_dim = twoE // 2
            assert twoE == 2 * expert_dim, "gate_up_proj last dim must be 2*expert_dim"
            assert down.shape == (num_experts, expert_dim, hidden)
            assert gate_up_bias.shape == (num_experts, 2 * expert_dim)
            assert down_bias.shape == (num_experts, hidden)

            # Build per-expert submodules exposing `gate_up_proj` and `down_proj` as nn.Linear
            class _GptOssExpertModule(nn.Module):
                def __init__(self, hidden_size: int, expert_dim: int):
                    super().__init__()
                    self.gate_up_proj = nn.Linear(hidden_size, 2 * expert_dim, bias=True)
                    self.down_proj = nn.Linear(expert_dim, hidden_size, bias=True)

            expert_modules = nn.ModuleList()
            for i in range(num_experts):
                mod = _GptOssExpertModule(hidden, expert_dim)
                with torch.no_grad():
                    mod.gate_up_proj.weight.copy_(gate_up[i].t())  # (hidden, 2E) -> Linear expects (out, in)
                    mod.gate_up_proj.bias.copy_(gate_up_bias[i])
                    mod.down_proj.weight.copy_(down[i].t())        # (E, hidden) -> (hidden, E) before .t()
                    mod.down_proj.bias.copy_(down_bias[i])
                expert_modules.append(mod)

            # Attach converted structure
            experts.experts = expert_modules

            # Override experts.forward to use the per-expert Linear modules
            def experts_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None):
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
                topk = router_indices.shape[1]
                device = hidden_states.device
                out = torch.zeros((hidden.shape[0], self.experts[0].down_proj.out_features), dtype=hidden_states.dtype, device=device)

                # accumulate per expert
                for expert_id, expert in enumerate(self.experts):
                    mask = (router_indices == expert_id).any(dim=-1)
                    if not mask.any():
                        continue
                    token_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    current = hidden[token_idx]
                    gate_up = expert.gate_up_proj(current)
                    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    gate = torch.clamp(gate, max=7.0)
                    up = torch.clamp(up, min=-7.0, max=7.0)
                    glu = gate * torch.sigmoid(gate * 1.702)
                    mid = (up + 1) * glu
                    down_out = expert.down_proj(mid)
                    weight = routing_weights[token_idx, (router_indices[token_idx] == expert_id).nonzero(as_tuple=True)[1]].unsqueeze(-1)
                    out[token_idx] += down_out * weight
                return out.view(batch_size, seq_len, -1)

            experts.forward = experts_forward.__get__(experts, type(experts))
