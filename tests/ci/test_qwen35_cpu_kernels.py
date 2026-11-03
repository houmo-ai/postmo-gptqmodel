"""Native GDN CPU execution stays valid when optional GPU kernels exist."""
import copy
import importlib

import pytest
import torch


def native_gdn(moe):
    name = "qwen3_5_moe" if moe else "qwen3_5"
    module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
    config_module = importlib.import_module(f"transformers.models.{name}.configuration_{name}")
    prefix = "Qwen3_5Moe" if moe else "Qwen3_5"
    config = getattr(config_module, prefix + "TextConfig")(
        hidden_size=128, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4)
    torch.manual_seed(871)
    gdn = getattr(module, prefix + "GatedDeltaNet")(config, layer_idx=0).half().eval()
    with torch.no_grad():
        for name, value in gdn.named_parameters():
            value.copy_(torch.ones_like(value) if "norm" in name else torch.randn_like(value) * .02)
    return gdn, module


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("offload_hook", [False, True], ids=["plain", "accelerate"])
def test_native_cpu_gdn_preserves_hf_bindings_and_state(moe, offload_hook):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from gptqmodel.models.definitions.qwen3_5_kernels import guard_qwen35_cpu_kernels

    torch.set_num_threads(2)
    model, native = native_gdn(moe)
    forward = type(model).forward
    kernel = native.torch_chunk_gated_delta_rule
    state = {name: value.clone() for name, value in model.state_dict().items()}
    if offload_hook:
        add_hook_to_module(model, AlignDevicesHook(execution_device="cpu"))
    guard_qwen35_cpu_kernels(model)
    guard_qwen35_cpu_kernels(model)  # Re-loading must not nest more wrappers.
    hidden = torch.randn(1, 16, 128).half() * .1
    with torch.inference_mode():
        output = model(hidden)
        assert output.shape == hidden.shape and torch.isfinite(output).all()
        torch.testing.assert_close(copy.deepcopy(model)(hidden), output)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert type(model).forward is forward
    assert native.torch_chunk_gated_delta_rule is kernel
