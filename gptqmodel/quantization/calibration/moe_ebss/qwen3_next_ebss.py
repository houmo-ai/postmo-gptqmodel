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
        # gate returns (router_logits, routing_weights, selected_experts)
        # routing_weights are already softmax + topk + norm applied inside the router
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        if self.save_expert_cnt:
            if (
                self.top_cnt is None
                or self.top_cnt.shape[0] != batch_size
                or self.top_cnt.device != hidden_states_reshaped.device
            ):
                self.top_cnt = torch.zeros(
                    batch_size,
                    self.gate.num_experts,
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

        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

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
