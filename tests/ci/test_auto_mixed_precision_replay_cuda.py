"""CUDA residency contract on a reduced-dimension real hybrid Qwen MoE."""
import pytest
import torch

from _qwen35_mixed_fixture import native_qwen35_cpu, samples, tiny_model
from gptqmodel.quantization.auto_mixed_precision import search_precision


@pytest.mark.usefixtures("native_qwen35_cpu")
@pytest.mark.parametrize("fail", [False, True])
def test_only_one_decoder_block_is_resident_and_cleanup(fail):
    assert torch.cuda.is_available(), "CUDA lane requires a GPU"
    model = tiny_model()
    layers = model.model.language_model.layers
    before = {n: t.clone() for n, t in model.state_dict().items()}
    calls = []
    def check(layer, args):
        resident = [m for m in layers if next(m.parameters()).device.type == "cuda"]
        assert resident == [layer]
        calls.append(layer)
        if fail and layer is layers[1]:
            raise RuntimeError("injected CUDA failure")
    handles = [layer.register_forward_pre_hook(check) for layer in layers]
    try:
        kwargs = dict(nsamples=2, seqlen=32, positions_per_sample=8, max_trials=2)
        if fail:
            with pytest.raises(RuntimeError, match="injected CUDA failure"):
                search_precision(model, samples(), kwargs, device="cuda:0")
        else:
            result = search_precision(model, samples(), kwargs, device="cuda:0")
            assert result["status"] == "accepted"
    finally:
        for handle in handles:
            handle.remove()
    if not fail:
        resident = search_precision(tiny_model(), samples(), dict(kwargs, execution_mode="resident"), device="cuda:0")
        assert result["unit_bits"] == resident["unit_bits"]
        for key in ("mean_kl", "delta_nll", "ppl_increase"):
            assert result["all_w8_metrics"][key] == pytest.approx(resident["all_w8_metrics"][key], abs=1e-7)
    assert calls and all(t.device.type == "cpu" for t in model.parameters())
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert model.model.language_model.layers is layers
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
