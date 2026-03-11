import builtins

import torch
import torch.nn.functional as F

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM, Qwen3NextSparseMoeBlock
except ImportError as exc:
    raise ImportError("Qwen3-Next EBSS requires transformers>=4.57.1.") from exc


def _get_num_beams():
    num_beams = getattr(builtins, "NUM_BEAMS", None)
    if not isinstance(num_beams, int) or num_beams <= 0:
        raise RuntimeError("builtins.NUM_BEAMS must be set to a positive integer before loading Qwen3-Next.")
    return num_beams


def patch_qwen3_next_sparse_moe_block():
    if getattr(Qwen3NextSparseMoeBlock, "_ebss_patched", False):
        return

    original_init = Qwen3NextSparseMoeBlock.__init__

    def patched_init(self, config):
        original_init(self, config)
        self.save_expert_cnt = False
        self.top_cnt = None

    def clear_expert_cnt(self):
        if self.top_cnt is not None:
            self.top_cnt.zero_()

    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_reshaped.dtype)

        if self.save_expert_cnt:
            if (
                self.top_cnt is None
                or self.top_cnt.shape[0] != batch_size
                or self.top_cnt.device != hidden_states_reshaped.device
            ):
                self.top_cnt = torch.zeros(
                    batch_size,
                    self.num_experts,
                    dtype=torch.float32,
                    device=hidden_states_reshaped.device,
                )
            flat_selected_experts = selected_experts.view(batch_size, -1)
            step_cnt = torch.zeros_like(self.top_cnt)
            step_cnt.scatter_add_(
                1,
                flat_selected_experts,
                torch.ones_like(flat_selected_experts, dtype=step_cnt.dtype),
            )
            self.top_cnt += step_cnt

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_reshaped.dtype,
            device=hidden_states_reshaped.device,
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_reshaped[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_reshaped.dtype))

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    Qwen3NextSparseMoeBlock.__init__ = patched_init
    Qwen3NextSparseMoeBlock.clear_expert_cnt = clear_expert_cnt
    Qwen3NextSparseMoeBlock.forward = patched_forward
    Qwen3NextSparseMoeBlock._ebss_patched = True


__all__ = ["Qwen3NextForCausalLM", "patch_qwen3_next_sparse_moe_block"]
