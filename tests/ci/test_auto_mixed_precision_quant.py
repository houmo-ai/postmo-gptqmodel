"""Real native hybrid-MoE search -> solver -> packed reload, including head W4/W5."""
import json
from pathlib import Path

import pytest
import torch

from _qwen35_mixed_fixture import save_image_dataset, save_source


@pytest.fixture(autouse=True)
def restore_loader_initializers(monkeypatch):
    """Packed reload disables global initializers; keep later fixtures isolated."""
    import transformers.modeling_utils as modeling_utils

    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(modeling_utils, "_init_weights",
                        getattr(modeling_utils, "_init_weights", True), raising=False)


@pytest.mark.parametrize("backend,initial", [("autoround", 5), ("gptq", 8)])
@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("head_bits", [4, 5])
def test_search_quantize_and_reload(backend, initial, with_images, head_bits, tmp_path):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.auto_mixed_precision import read_precision_plan
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    source = save_source(tmp_path / "source")
    calibration = tmp_path / "calibration.jsonl"
    text = " ".join(f"t{i}" for i in range(3, 100))
    calibration.write_text("\n".join(json.dumps({"text": text}) for _ in range(4)) + "\n")
    output = tmp_path / "output"
    common = dict(model_dir=str(source), output_dir=str(output), topology="moe",
                  device="cuda:0", group_size=64, bits=4, nsamples=2, seqlen=32,
                  expert_bits=4, expert_down_bits=4,
                  head={"bits": head_bits, "iters": 2},
                  auto_mixed_precision={"initial_bits": initial, "max_trials": 2,
                      "nsamples": 2, "seqlen": 32, "positions_per_sample": 8,
                      "max_kl": 1, "max_ppl_increase": 1, "device": "cuda:0"})
    if with_images:
        common["auto_mixed_precision"].update(
            image_dataset=str(save_image_dataset(tmp_path / "images")),
            image_nsamples=2, image_max_pixels=4096, image_seqlen=128)
    if backend == "autoround":
        result = quantize_qwen35_autoround(**common, dataset=str(calibration), iters=2,
            batch_size=1, data_type="int_sym_xh", format="auto_round:gptqmodel", device_map="0")
    else:
        result = quantize_qwen35(**common, calibration_jsonl=str(calibration), device_map=None,
            batch_size=1, check_quant_vision_demo=False, skip_inference=True)
    assert Path(result.output_dir) == output
    plan = read_precision_plan(output)
    assert result.provenance["auto_mixed_precision"] == plan
    assert Path(str(output) + ".auto_mixed_precision_search.json").is_file()
    assert set(plan["module_bits"].values()) == ({4, 5} if initial == 5 else {5, 8})
    report = json.loads((output / "auto_mixed_precision.json").read_text())
    assert report["status"] == "accepted" and report["final_quantization_validated"] is False
    if with_images:
        assert report["modality_sample_counts"]["image"] == 2
        assert report["selected_metrics"]["modalities"]["image"]["tokens"] == 2
    config = json.loads((output / "config.json").read_text())
    assert not config["tie_word_embeddings"]
    loaded = GPTQModel.load(str(output), device="cpu", backend="torch", torch_dtype=torch.float16)
    modules = dict(loaded.model.named_modules())
    for name, bits in plan["module_bits"].items():
        module = modules[name]
        assert module.logical_bits == bits
        assert module.bits == (8 if bits == 5 else bits)
        if bits == 5:
            codes = (module.qweight.to(torch.int64).unsqueeze(-1) >>
                     torch.tensor([0, 8, 16, 24])) & 255
            assert codes.max() <= 31
    head = modules["lm_head"]
    assert head.logical_bits == head_bits
    assert head.bits == (8 if head_bits == 5 else head_bits)
    if head_bits == 5:
        codes = (head.qweight.to(torch.int64).unsqueeze(-1) >> torch.tensor([0, 8, 16, 24])) & 255
        assert codes.max() <= 31
    assert result.provenance["head_stage"]["head"]["algorithm"] == backend
    assert result.provenance["head_stage"]["head"]["bits"] == head_bits
    assert not Path(str(output) + ".backbone").exists()
    assert not list(tmp_path.glob("output.backbone-*"))
    experts = [m for n, m in modules.items() if ".experts." in n and hasattr(m, "qweight")]
    assert experts and all(m.logical_bits == 4 for m in experts)
    # Native generation forward on CUDA exercises actual loaded W4/W5/W8 kernels.
    loaded.model.to("cuda:0").eval()
    with torch.inference_mode():
        logits = loaded.model(input_ids=torch.tensor([[3, 4, 5, 6]], device="cuda:0"),
                              use_cache=False).logits
    assert torch.isfinite(logits).all() and logits.shape == (1, 4, 256)
