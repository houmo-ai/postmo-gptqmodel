"""Real hybrid MoE forwards: replay equivalence and transactional cleanup."""
import pytest
import torch

from _qwen35_mixed_fixture import native_qwen35_cpu, samples, save_source, save_image_dataset, tiny_model
from gptqmodel.quantization.auto_mixed_precision import (
    load_image_records, parse_auto_mixed_precision, search_precision,
)


@pytest.mark.usefixtures("native_qwen35_cpu")
@pytest.mark.parametrize("with_images", [False, True])
def test_layerwise_matches_resident_search(tmp_path, with_images):
    options = dict(nsamples=2, seqlen=32, positions_per_sample=8, max_trials=3)
    images = None
    if with_images:
        source = tmp_path / "source"
        save_source(source)
        options.update(image_dataset=str(save_image_dataset(tmp_path)), image_nsamples=2,
                       image_max_pixels=4096, image_seqlen=128)
        images = load_image_records(source, parse_auto_mixed_precision(options))
    resident = search_precision(tiny_model(), samples(), dict(options, execution_mode="resident"),
                                image_samples=images, device="cpu")
    replayed = search_precision(tiny_model(), samples(), options, image_samples=images, device="cpu")
    for key in ("unit_bits", "module_bits", "selected_metrics", "all_w8_metrics", "trials",
                "reconstruction_errors", "data_sha256", "target_weight_sha256"):
        assert replayed[key] == resident[key], key


@pytest.mark.usefixtures("native_qwen35_cpu")
@pytest.mark.parametrize("stage", ["prepare", "replay", "candidate"])
def test_failed_replay_restores_model(stage):
    model = tiny_model()
    before = {n: t.clone() for n, t in model.state_dict().items()}
    layers = model.model.language_model.layers
    training = [m.training for m in model.modules()]
    model.model.rope_deltas = torch.tensor([[9]])
    module = model.model.language_model.embed_tokens if stage == "prepare" else layers[1]
    calls = []
    def fail(*args):
        calls.append(1)
        if stage != "candidate" or len(calls) > 2:
            raise RuntimeError("injected replay failure")
    hook = module.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected replay failure"):
            search_precision(model, samples(), dict(nsamples=2, seqlen=32, positions_per_sample=8), device="cpu")
    finally:
        hook.remove()
    assert model.model.language_model.layers is layers
    assert [m.training for m in model.modules()] == training
    assert model.model.rope_deltas.item() == 9
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert all(not m._forward_pre_hooks for m in model.modules())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_execution_mode_validation():
    assert parse_auto_mixed_precision(True).execution_mode == "layerwise"
    for mode in ("auto", None, True):
        with pytest.raises(ValueError, match="execution_mode"):
            parse_auto_mixed_precision({"execution_mode": mode})


@pytest.mark.usefixtures("native_qwen35_cpu")
def test_text_only_causal_lm_replay():
    from transformers import Qwen3_5MoeForCausalLM
    full = tiny_model()
    model = Qwen3_5MoeForCausalLM(full.config.text_config)
    model.model, model.lm_head = full.model.language_model, full.lm_head
    options = dict(nsamples=2, seqlen=32, positions_per_sample=8, max_trials=2)
    resident = search_precision(model, samples(), dict(options, execution_mode="resident"), device="cpu")
    replayed = search_precision(model, samples(), options, device="cpu")
    assert replayed["trials"] == resident["trials"]
    assert model.lm_head.weight is model.get_input_embeddings().weight
