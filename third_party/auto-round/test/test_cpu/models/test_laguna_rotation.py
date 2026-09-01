from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts_laguna import chat_quantized, quantize_laguna
from scripts_laguna.rotation_utils import (
    _random_residual_hadamard,
    apply_quarot,
    validate_laguna_quarot,
)


class TinyRMSNorm(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(size, dtype=torch.float64).uniform_(0.7, 1.3))

    def forward(self, inputs):
        variance = inputs.pow(2).mean(dim=-1, keepdim=True)
        return inputs * torch.rsqrt(variance + 1e-12) * self.weight


class TinyAttention(torch.nn.Module):
    def __init__(self, hidden_size=4, num_heads=2, head_dim=2, gate_per_head=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate_per_head = gate_per_head
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float64)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float64)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float64)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=torch.float64)
        gate_width = num_heads if gate_per_head else num_heads * head_dim
        self.g_proj = torch.nn.Linear(hidden_size, gate_width, bias=False, dtype=torch.float64)

    def forward(self, hidden_states):
        batch, seqlen, _ = hidden_states.shape

        def heads(projection):
            return projection(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        query = heads(self.q_proj)
        key = heads(self.k_proj)
        value = heads(self.v_proj)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.head_dim**0.5
        output = torch.matmul(torch.softmax(scores, dim=-1), value)
        output = output.transpose(1, 2).reshape(batch, seqlen, -1)
        gate = F.softplus(self.g_proj(hidden_states))
        if self.gate_per_head:
            output = (
                output.view(batch, seqlen, self.num_heads, self.head_dim) * gate.unsqueeze(-1)
            ).reshape(batch, seqlen, -1)
        else:
            output = output * gate
        return self.o_proj(output)


class TinyDenseMLP(torch.nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=6):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float64)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float64)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float64)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TinyRouter(torch.nn.Module):
    def __init__(self, num_experts=2, hidden_size=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden_size, dtype=torch.float64))

    def forward(self, hidden_states):
        scores = torch.sigmoid(F.linear(hidden_states, self.weight))
        return scores.max(dim=-1)


class TinyExperts(torch.nn.Module):
    def __init__(self, num_experts=2, hidden_size=4, intermediate_size=3):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float64)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float64)
        )

    def forward(self, hidden_states, selected_experts, routing_weights):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        selected = selected_experts.reshape(-1)
        weights = routing_weights.reshape(-1)
        output = torch.zeros_like(flat)
        for token_index in range(flat.shape[0]):
            expert_index = int(selected[token_index])
            gate, up = F.linear(flat[token_index], self.gate_up_proj[expert_index]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[expert_index])
            output[token_index] = expert_output * weights[token_index]
        return output.reshape_as(hidden_states)


class TinySparseMLP(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.gate = TinyRouter(hidden_size=hidden_size)
        self.experts = TinyExperts(hidden_size=hidden_size)
        self.shared_expert = TinyDenseMLP(hidden_size=hidden_size, intermediate_size=3)

    def forward(self, hidden_states):
        routing_weights, selected_experts = self.gate(hidden_states)
        return self.shared_expert(hidden_states) + self.experts(
            hidden_states,
            selected_experts,
            routing_weights,
        )


class TinyLayer(torch.nn.Module):
    def __init__(self, mlp, *, gate_per_head=True):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(4)
        self.post_attention_layernorm = TinyRMSNorm(4)
        self.self_attn = TinyAttention(gate_per_head=gate_per_head)
        self.mlp = mlp

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyBackbone(torch.nn.Module):
    def __init__(self, *, gate_per_head=True):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 4, dtype=torch.float64)
        self.layers = torch.nn.ModuleList(
            [
                TinyLayer(TinyDenseMLP(), gate_per_head=gate_per_head),
                TinyLayer(TinySparseMLP(), gate_per_head=gate_per_head),
            ]
        )
        self.norm = TinyRMSNorm(4)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class TinyLaguna(torch.nn.Module):
    def __init__(self, *, gate_per_head=True):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, model_type="laguna")
        self.model = TinyBackbone(gate_per_head=gate_per_head)
        self.lm_head = torch.nn.Linear(4, 11, bias=False, dtype=torch.float64)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def test_laguna_quarot_preserves_dense_sparse_router_and_gated_attention_outputs():
    torch.manual_seed(7)
    model = TinyLaguna().eval()
    input_ids = torch.tensor([[1, 4, 2], [3, 8, 5]])
    expected = model(input_ids).detach()
    router_before = model.model.layers[1].mlp.gate(
        model.model.layers[1].post_attention_layernorm(
            model.model.layers[0](model.model.embed_tokens(input_ids))
        )
    ).indices

    apply_quarot(
        model,
        device="cpu",
        seed=11,
        headwise_v_o=True,
        chunk_rows=2,
        chunk_columns=2,
        matrix_batch=1,
        compute_dtype=torch.float64,
    )
    actual = model(input_ids).detach()

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert model.config.quarot_config["applied"] is True
    assert model.config.quarot_config["headwise_v_o"] is True
    for layer in model.model.layers:
        assert torch.equal(layer.input_layernorm.weight, torch.ones_like(layer.input_layernorm.weight))
        assert torch.equal(layer.post_attention_layernorm.weight, torch.ones_like(layer.post_attention_layernorm.weight))
    assert torch.equal(model.model.norm.weight, torch.ones_like(model.model.norm.weight))

    # An exactly equivalent FP rotation must preserve the discrete expert route.
    dense_hidden = model.model.layers[0](model.model.embed_tokens(input_ids))
    router_after = model.model.layers[1].mlp.gate(
        model.model.layers[1].post_attention_layernorm(dense_hidden)
    ).indices
    assert torch.equal(router_after, router_before)


def test_headwise_rotation_rejects_per_element_gate_before_mutation():
    torch.manual_seed(3)
    model = TinyLaguna(gate_per_head=False)
    embedding_before = model.model.embed_tokens.weight.detach().clone()

    with pytest.raises(ValueError, match="per-element gating"):
        validate_laguna_quarot(model, headwise_v_o=True)

    assert torch.equal(model.model.embed_tokens.weight, embedding_before)


def test_chat_input_device_accepts_torch_device_in_accelerate_map():
    model = SimpleNamespace(hf_device_map={"model.embed_tokens": torch.device("cuda:2")})

    assert chat_quantized.input_device(model) == torch.device("cuda:2")


def test_composite_residual_hadamard_is_orthonormal():
    torch.manual_seed(13)
    matrix = _random_residual_hadamard(
        24,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert torch.allclose(matrix @ matrix.T, torch.eye(24, dtype=torch.float64), atol=1e-6)


def test_quantize_pipeline_passes_rotated_model_object_to_autoround(monkeypatch, tmp_path):
    rotated_model = object()
    tokenizer = object()
    captured = {}

    args = SimpleNamespace(
        model="/models/laguna",
        output_dir=str(tmp_path / "quantized"),
        expert_bits=4,
        rest_bits=8,
        group_size=64,
        sym=True,
        iters=1,
        nsamples=1,
        seqlen=8,
        batch_size=1,
        gradient_accumulate_steps=1,
        dataset="unused",
        device_map="0",
        format="auto_round",
        seed=42,
        rotation="hadamard",
        rotation_device="cpu",
        rotation_headwise_v_o=True,
        rotation_chunk_rows=2,
        rotation_chunk_columns=2,
        rotation_matrix_batch=1,
        rotation_compute_dtype="float64",
        save_rotated=None,
        rotation_only=False,
        quant_lm_head=False,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        inspect_only=False,
    )

    class FakeAutoRound:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def quantize_and_save(self, **kwargs):
            captured["save"] = kwargs

    import auto_round

    monkeypatch.setattr(quantize_laguna, "parse_args", lambda: args)
    monkeypatch.setattr(
        quantize_laguna,
        "validate_environment",
        lambda _: SimpleNamespace(
            num_hidden_layers=2,
            num_experts=2,
            num_experts_per_tok=1,
            hidden_size=4,
            moe_intermediate_size=3,
        ),
    )
    monkeypatch.setattr(quantize_laguna, "load_and_rotate_model", lambda _: (rotated_model, tokenizer))
    monkeypatch.setattr(auto_round, "AutoRound", FakeAutoRound)

    quantize_laguna.main()

    assert captured["model"] is rotated_model
    assert captured["tokenizer"] is tokenizer
    assert captured["save"] == {"output_dir": args.output_dir, "format": "auto_round"}
