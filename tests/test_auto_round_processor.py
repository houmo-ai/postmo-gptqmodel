import torch
from types import SimpleNamespace
import pytest

from gptqmodel.looper.auto_round_processor import AutoRoundProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.converter import Qwen3_5MoeExpertsNew
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AutoRoundVersion, QuantizeConfig


def _prepare_dataset_func(**kwargs):
    return kwargs["calibration_dataset"]


def _make_processor(hessian_mse: bool = False, version: AutoRoundVersion = AutoRoundVersion.V1) -> AutoRoundProcessor:
    qcfg = QuantizeConfig(auto_round_enabled=True, 
        bits=4,
        group_size=4,
        sym=True,
        format=FORMAT.GPTQ,
        desc_act=False,
        iters=2,
        lr=0.05,
        minmax_lr=0.05,
        hessian_mse=hessian_mse,
        device="cpu",
        auto_round_version=version,
    )
    return AutoRoundProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[{"input_ids": torch.tensor([1, 2, 3, 4])}],
        prepare_dataset_func=_prepare_dataset_func,
        calibration_concat_size=None,
        calibration_sort=None,
        calibration_concat_separator=None,
        batch_size=1,
    )


@pytest.mark.parametrize("version", [AutoRoundVersion.V1, AutoRoundVersion.V2])
def test_auto_round_processor_quantizes_linear_and_emits_pack_qparams(version):
    processor = _make_processor(version=version)
    linear = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-1.0, 1.0, steps=linear.weight.numel(), dtype=linear.weight.dtype).reshape_as(linear.weight))
    named = NamedModule(linear, name="proj", full_name="layer.proj", layer_index=0)

    processor.preprocess(named)

    inputs = torch.randn(2, 8)
    outputs = linear(inputs)
    processor.pre_process_fwd_hook("proj")(linear, (inputs,), outputs)

    assert processor.has_captured_input_ids("proj") is True

    original_weight = linear.weight.detach().clone()
    processor.process(named, device=torch.device("cpu"))

    assert "q_scales" in named.state
    assert "q_zeros" in named.state
    assert "q_g_idx" in named.state
    assert named.state["q_scales"].shape == (4, 2)
    assert named.state["q_zeros"].shape == (4, 2)
    assert named.state["q_g_idx"].shape == (8,)
    assert not torch.equal(original_weight, linear.weight.detach())
    assert processor.qcfg.auto_round_version == version


def test_auto_round_processor_supports_hessian_mse_scale_init():
    processor = _make_processor(hessian_mse=True)
    linear = torch.nn.Linear(8, 4, bias=False)
    named = NamedModule(linear, name="proj", full_name="layer.proj", layer_index=0)

    processor.preprocess(named)

    inputs = torch.randn(3, 8)
    outputs = linear(inputs)
    processor.pre_process_fwd_hook("proj")(linear, (inputs,), outputs)
    processor.process(named, device=torch.device("cpu"))

    assert torch.all(named.state["q_scales"] > 0)
    assert torch.equal(named.state["q_g_idx"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32))


def test_auto_round_processor_quantizes_moe_expert_projection():
    processor = _make_processor()
    config = SimpleNamespace(
        hidden_size=8,
        moe_intermediate_size=16,
        num_experts=2,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )
    experts = Qwen3_5MoeExpertsNew(config)
    expert_proj = experts[0].gate_proj
    named = NamedModule(
        expert_proj,
        name="gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )

    processor.preprocess(named)

    inputs = torch.randn(2, config.hidden_size)
    outputs = expert_proj(inputs)
    processor.pre_process_fwd_hook("gate_proj")(expert_proj, (inputs,), outputs)
    processor.process(named, device=torch.device("cpu"))

    assert named.state["q_scales"].shape[0] == config.moe_intermediate_size
    assert named.state["q_g_idx"].numel() == config.hidden_size
    assert torch.isfinite(expert_proj.weight).all()


def _fake_quantize_from_round_value(weight, state):
    """Test quantizer whose serialized weight exposes the optimizer state."""
    rows, cols = weight.shape
    groups = (cols + state.qcfg.group_size - 1) // state.qcfg.group_size
    qdq_weight = state.value.to(weight.dtype)
    scales = torch.ones((rows, groups), dtype=torch.float32, device=weight.device)
    zeros = torch.full_like(scales, 7.5)
    g_idx = torch.arange(cols, dtype=torch.int32, device=weight.device) // state.qcfg.group_size
    return qdq_weight, scales, zeros, g_idx


def _prepare_single_step_processor(monkeypatch):
    processor = _make_processor(version=AutoRoundVersion.V1)
    processor.qcfg.iters = 1
    processor.qcfg.lr = 10.0
    processor.qcfg.minmax_lr = 10.0
    linear = torch.nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0 / 8.0)
    named = NamedModule(linear, name="proj", full_name="layer.proj", layer_index=0)
    processor.preprocess(named)
    inputs = torch.ones(1, 8)
    processor.pre_process_fwd_hook("proj")(linear, (inputs,), linear(inputs))
    monkeypatch.setattr(
        "gptqmodel.looper.auto_round_processor.auto_round_fake_quantize",
        _fake_quantize_from_round_value,
    )
    return processor, linear, named


def test_auto_round_single_step_does_not_save_unevaluated_worse_state(monkeypatch):
    processor, linear, named = _prepare_single_step_processor(monkeypatch)

    processor.process(named, device=torch.device("cpu"))

    # Initial state predicts zero (loss 1). Adam(lr=10) moves the eight round
    # values to about 10, whose prediction/loss is vastly worse. The old code
    # assigned loss 1 to that post-step state and serialized it.
    assert torch.equal(linear.weight, torch.zeros_like(linear.weight))
    assert processor.log[-1]["auto_round_best_step"] == "0"
    assert processor.log[-1]["loss"] == "1.0000000000"


def test_auto_round_non_finite_update_restores_last_evaluated_state(monkeypatch):
    processor, linear, named = _prepare_single_step_processor(monkeypatch)
    processor.tasks["proj"].qcfg.not_use_best_mse = True

    def non_finite_after_update(weight, state):
        result = list(_fake_quantize_from_round_value(weight, state))
        if torch.count_nonzero(state.value.detach()).item():
            result[0] = result[0] * float("nan")
        return tuple(result)

    monkeypatch.setattr(
        "gptqmodel.looper.auto_round_processor.auto_round_fake_quantize",
        non_finite_after_update,
    )
    processor.process(named, device=torch.device("cpu"))

    assert torch.isfinite(linear.weight).all()
    assert torch.equal(linear.weight, torch.zeros_like(linear.weight))
    assert processor.log[-1]["auto_round_fallback"] == "non_finite_loss"
    assert processor.log[-1]["loss"] == "failsafe(rtn): 1.0000000000"


def test_auto_round_not_use_best_mse_saves_evaluated_final_state(monkeypatch):
    processor, linear, named = _prepare_single_step_processor(monkeypatch)
    processor.tasks["proj"].qcfg.not_use_best_mse = True

    processor.process(named, device=torch.device("cpu"))

    assert torch.count_nonzero(linear.weight).item() == linear.weight.numel()
    assert processor.log[-1]["auto_round_best_step"] == "1"
    reference = torch.tensor([[1.0]])
    selected_loss = torch.mean((torch.nn.functional.linear(torch.ones(1, 8), linear.weight) - reference) ** 2)
    assert float(processor.log[-1]["loss"]) == pytest.approx(selected_loss.item())


def test_auto_round_low_sample_failsafe_skips_optimizer(monkeypatch):
    processor, linear, named = _prepare_single_step_processor(monkeypatch)
    processor.tasks["proj"].failsafe.threshold = "50%"

    processor.process(named, device=torch.device("cpu"))

    # Calibration has four expected tokens but only one captured row.
    assert torch.equal(linear.weight, torch.zeros_like(linear.weight))
    assert processor.log[-1]["samples"] == "1"
    assert processor.log[-1]["auto_round_fallback"] == "insufficient_samples"
    assert processor.log[-1]["loss"] == "failsafe(rtn): 1.0000000000"


def test_auto_round_unactivated_expert_uses_weight_only_failsafe(monkeypatch):
    processor = _make_processor(version=AutoRoundVersion.V1)
    linear = torch.nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0 / 8.0)
    named = NamedModule(
        linear,
        name="down_proj",
        full_name="model.layers.0.mlp.experts.7.down_proj",
        layer_index=0,
    )
    processor.preprocess(named)
    monkeypatch.setattr(
        "gptqmodel.looper.auto_round_processor.auto_round_fake_quantize",
        _fake_quantize_from_round_value,
    )

    processor.process(named, device=torch.device("cpu"))

    assert torch.equal(linear.weight, torch.zeros_like(linear.weight))
    assert processor.log[-1]["samples"] == "0"
    assert processor.log[-1]["auto_round_fallback"] == "insufficient_samples"
    assert processor.log[-1]["loss"] == "failsafe(rtn): 0.0156250000"
