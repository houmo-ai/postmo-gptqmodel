"""Moving GDN between GPU and CPU preserves native accelerator dispatch."""
import pytest
import torch

from test_qwen35_cpu_kernels import native_gdn


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_cpu_reference_and_original_gpu_kernels_agree(moe):
    from gptqmodel.models.definitions.qwen3_5_kernels import guard_qwen35_cpu_kernels

    torch.set_num_threads(2)
    model, native = native_gdn(moe)
    kernel = native.torch_chunk_gated_delta_rule
    model.cuda()
    hidden = torch.randn(1, 16, 128, device="cuda").half() * .1
    with torch.inference_mode():
        expected = model(hidden)
        guard_qwen35_cpu_kernels(model)
        torch.testing.assert_close(model(hidden), expected)
        cpu = model.cpu()(hidden.cpu())
        assert torch.isfinite(cpu).all()
        torch.testing.assert_close(cpu.float(), expected.cpu().float(), rtol=.02, atol=.002)
        torch.testing.assert_close(model.cuda()(hidden), expected)
    assert native.torch_chunk_gated_delta_rule is kernel
