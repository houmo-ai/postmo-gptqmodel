"""AutoRound GLM precision, native expert semantics and calibration contracts."""

from types import SimpleNamespace

import pytest
import torch

from _glm53_fixture import write_tiny_glm53
from gptqmodel.recipes.glm5_3_flash_autoround import (
    build_glm53_autoround_calibration,
    build_glm53_autoround_policy,
    install_glm53_autoround_loss_guard,
    install_glm53_autoround_expert_checkpointing,
    quantize_glm5_3_flash_autoround,
)


def test_autoround_native_clamped_experts_and_policy(tmp_path):
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        prepare_model_for_moe_quantization,
    )
    from auto_round.utils import to_standard_regex
    import re

    model = write_tiny_glm53(tmp_path / "source")
    moe = model.model.language_model.layers[1].mlp
    inputs = torch.randn(2, 7, 128) * 40
    expected = moe(inputs).detach()
    assert prepare_model_for_moe_quantization(model) == [
        "model.language_model.layers.1.mlp.experts"
    ]
    actual = moe(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    actual.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in moe.experts.parameters()
    )
    policy = build_glm53_autoround_policy(model)
    pattern = next(key for key in policy if key.startswith("^"))
    assert (
        sum(
            bool(re.match(to_standard_regex(pattern), name))
            for name, _ in model.named_modules()
        )
        == 12
    )
    for name in ("forget_gate.f_a_proj", "forget_gate.f_b_proj", "b_proj"):
        assert policy[f"model.language_model.layers.0.self_attn.{name}"]["bits"] == 16
    assert policy["model.language_model.layers.1.self_attn.kv_b_proj"]["bits"] == 16


def test_autoround_calibration_records_only_effective_full_rows(tmp_path):
    from transformers import AutoTokenizer

    write_tiny_glm53(tmp_path / "source")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "source")
    valid = " ".join(f"word{i}" for i in range(32))
    rows, evidence = build_glm53_autoround_calibration(
        tokenizer,
        data=["word1", "word1 " * 32, valid, valid],
        nsamples=2,
        seqlen=32,
    )
    assert len(rows) == 2 and evidence["actual_tokens"] == 64
    _, repeated = build_glm53_autoround_calibration(
        tokenizer,
        data=["word1", "word1 " * 32, valid, valid],
        nsamples=2,
        seqlen=32,
    )
    assert evidence == repeated
    with pytest.raises(ValueError, match="full-length"):
        build_glm53_autoround_calibration(
            tokenizer, data=["word1"], nsamples=1, seqlen=32
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_autoround_aborts_nonfinite_loss(value):
    ar = SimpleNamespace(_get_loss=lambda: torch.tensor(value))
    evidence = install_glm53_autoround_loss_guard(ar)
    with pytest.raises(FloatingPointError, match="publication aborted"):
        ar._get_loss()
    assert evidence["loss_evaluations"] == 0


@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_expert_recomputation_preserves_gradients_and_reduces_saved_tensors(
    input_requires_grad,
):
    import copy
    from auto_round.wrapper import WrapperLinear

    torch.manual_seed(489)
    linear = torch.nn.Linear(128, 64, bias=False).requires_grad_(False)
    for name, value in {
        "bits": 4,
        "group_size": 64,
        "sym": True,
        "act_bits": 16,
        "data_type": "int_sym_gptq",
        "scale_dtype": torch.float16,
    }.items():
        setattr(linear, name, value)
    baseline = WrapperLinear(linear)
    wrapped = copy.deepcopy(baseline)
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.experts = torch.nn.ModuleList([torch.nn.Module()])
    block.mlp.experts[0].gate_proj = wrapped
    # Wrappers already exist: the intercepted setup is irrelevant to this
    # differential test of actual QDQ forward/backward and saved tensors.
    ar = SimpleNamespace(wrapper_block=lambda *args, **kwargs: ([], []))
    evidence = install_glm53_autoround_expert_checkpointing(ar)
    ar.wrapper_block(block)
    assert evidence["checkpointed_projections"] == 1
    values = torch.randn(5, 128)

    def run(module):
        inputs = values.clone().requires_grad_(input_requires_grad)
        saved = []

        def pack(value):
            saved.append(value.numel())
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = module(inputs)
        output.square().mean().backward()
        gradients = {
            name: param.grad.clone()
            for name, param in module.named_parameters()
            if param.grad is not None
        }
        return output, gradients, sum(saved), inputs.grad

    reference, expected_gradients, original_saved, expected_input_gradient = run(
        baseline
    )
    actual, gradients, saved, input_gradient = run(wrapped)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert (
        gradients.keys() == expected_gradients.keys()
        and gradients["value"].abs().sum() > 0
    )
    for name in gradients:
        torch.testing.assert_close(
            gradients[name], expected_gradients[name], rtol=0, atol=0
        )
    if input_requires_grad:
        torch.testing.assert_close(
            input_gradient, expected_input_gradient, rtol=0, atol=0
        )
    assert saved < original_saved // 4


@pytest.mark.parametrize(
    "override",
    [
        {"method": "gptq"},
        {"iters": 0},
        {"expert_bits": 8},
        {"batch_size": 3},
        {"nsamples": 4, "gradient_accumulate_steps": 8},
    ],
)
def test_autoround_rejects_unsupported_requests(tmp_path, override):
    with pytest.raises(ValueError):
        quantize_glm5_3_flash_autoround(
            model_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            **override,
        )
