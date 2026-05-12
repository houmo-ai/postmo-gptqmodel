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
