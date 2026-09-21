"""Real GPTQ, packed save/load and multimodal checkpoint retention."""

from pathlib import Path
import subprocess
import sys


def _quantize_and_reload(tmp_path):
    import json
    import pytest
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from _glm53_fixture import calibration_texts, write_tiny_glm53
    from gptqmodel import GPTQModel
    from gptqmodel.recipes.glm5_3_flash import (
        quantize_glm5_3_flash,
        audit_glm5_3_flash_quantized_artifact,
    )
    from gptqmodel.utils import BACKEND

    source, output = tmp_path / "source", tmp_path / "quantized"
    write_tiny_glm53(source)
    result = quantize_glm5_3_flash(
        model_dir=str(source),
        output_dir=str(output),
        calibration_data=calibration_texts(),
        calibration_dataset=None,
        nsamples=16,
        seqlen=32,
        device="cuda:0",
        moe_routing_batch_size=2,
    )
    assert result.output_dir == str(output)
    assert audit_glm5_3_flash_quantized_artifact(output)["w4_modules"] == 12
    model = GPTQModel.load(str(output), backend=BACKEND.TORCH, device="cuda:0")
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], device="cuda:0")
    with torch.no_grad():
        actual = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits
    assert torch.isfinite(actual).all()
    # Preserve the vision tower even when calibration rows are text-only.
    assert model.model.model.visual.blocks[0].attn.qkv.weight.shape == (192, 64)
    with safe_open(output / "model.safetensors", framework="pt") as reader:
        prefix = "model.language_model.layers.1.mlp.experts.0.gate_proj"
        torch.testing.assert_close(
            model.model.get_submodule(prefix).qweight.cpu(),
            reader.get_tensor(prefix + ".qweight"),
        )
    # Corrupt real solver output without changing any shapes or dtypes. Audit
    # must reject invalid values before a large export can start.
    weights = load_file(output / "model.safetensors")
    for suffix, bad_value, message in (
        ("scales", float("nan"), "Invalid GPTQ scales"),
        ("scales", 0.0, "Invalid GPTQ scales"),
        ("g_idx", 1, "consecutive G64 groups"),
    ):
        key = prefix + "." + suffix
        invalid = dict(weights)
        invalid[key] = weights[key].clone()
        invalid[key].view(-1)[0] = bad_value
        temporary = output / "invalid.safetensors"
        save_file(invalid, temporary)
        temporary.replace(output / "model.safetensors")
        with pytest.raises(ValueError, match=message):
            audit_glm5_3_flash_quantized_artifact(output)
    temporary = output / "restored.safetensors"
    save_file(weights, temporary)
    temporary.replace(output / "model.safetensors")
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["desc_act"] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="desc_act=False"):
        audit_glm5_3_flash_quantized_artifact(output)


def test_hybrid_gptq_save_reload(tmp_path):
    # Match the production worker boundary. GPTQ's loader/Defuser alter HF and
    # torch initialization globals; a real subprocess contains all such state
    # and releases GPU allocations before the next model's regression.
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    assert process.returncode == 0, process.stdout


if __name__ == "__main__":
    _quantize_and_reload(Path(sys.argv[1]))
