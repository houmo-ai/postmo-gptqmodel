"""Actual AutoRound block tuning, immediate packed save and fresh-process reload."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def _produce(root, iters=4):
    from dataclasses import asdict
    from _glm53_fixture import calibration_texts, write_tiny_glm53
    from gptqmodel.recipes.glm5_3_flash_autoround import quantize_glm5_3_flash_autoround

    write_tiny_glm53(root / "source")
    generation_path = root / "source" / "generation_config.json"
    generation = json.loads(generation_path.read_text())
    generation.update(do_sample=False, top_p=0.95)
    generation_path.write_text(json.dumps(generation))
    result = quantize_glm5_3_flash_autoround(
        model_dir=str(root / "source"),
        output_dir=str(root / "quantized"),
        calibration_data=calibration_texts(),
        calibration_dataset=None,
        nsamples=16,
        seqlen=32,
        batch_size=2,
        gradient_accumulate_steps=1,
        iters=iters,
        device_map="0",
    )
    assert (
        result.method == "autoround"
        and result.provenance["tuning"]["loss_evaluations"] >= 2 * iters
    )
    assert result.provenance["expert_checkpointing"]["checkpointed_projections"] == 12
    saved_generation = json.loads(
        (root / "quantized" / "generation_config.json").read_text()
    )
    assert saved_generation["do_sample"] is True and saved_generation["top_p"] == 0.95
    assert json.loads(generation_path.read_text())["do_sample"] is False
    (root / "result.json").write_text(json.dumps(asdict(result)))


def _reload(root, iters=4):
    import torch
    from safetensors import safe_open
    from gptqmodel import GPTQModel
    from gptqmodel.utils import BACKEND
    from gptqmodel.recipes.glm5_3_flash import audit_glm5_3_flash_quantized_artifact

    checkpoint = root / "quantized"
    audit = audit_glm5_3_flash_quantized_artifact(checkpoint)
    assert (audit["layers"], audit["w4_modules"], audit["w8_modules"]) == (2, 12, 16)
    config = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
    assert config["provider"] == "auto-round" and config["iters"] == iters
    model = GPTQModel.load(str(checkpoint), backend=BACKEND.TORCH, device="cuda:0")
    ids = torch.arange(3, 19, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits
    assert torch.isfinite(output).all()
    assert model.model.model.visual.blocks[0].attn.qkv.weight.shape == (192, 64)
    index = checkpoint / "model.safetensors.index.json"
    prefix = "model.language_model.layers.1.mlp.experts.0.gate_proj"
    shard = (
        json.loads(index.read_text())["weight_map"][prefix + ".qweight"]
        if index.exists()
        else "model.safetensors"
    )
    with safe_open(checkpoint / shard, framework="pt") as reader:
        torch.testing.assert_close(
            model.model.get_submodule(prefix).qweight.cpu(),
            reader.get_tensor(prefix + ".qweight"),
        )


@pytest.mark.parametrize("iters", [2, 200])
def test_autoround_hybrid_save_fresh_reload(tmp_path, iters):
    for phase in ("produce", "reload"):
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), phase, str(tmp_path), str(iters)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert run.returncode == 0, run.stdout + run.stderr


def _produce_failure(root, iters=2):
    from safetensors import safe_open
    import gptqmodel.recipes.glm5_3_flash_autoround as recipe

    def fail_audit(*args, **kwargs):
        raise ValueError("injected post-save audit failure")

    # Inject a storage lifecycle failure after real native tuning and saving.
    recipe.canonicalize_glm53_autoround_checkpoint = fail_audit
    with pytest.raises(ValueError, match="injected post-save") as caught:
        _produce(root, iters)
    assert not (root / "quantized").exists()
    staging = list(root.glob(".quantized.autoround-*"))
    assert len(staging) == 1
    assert str(staging[0]) in " ".join(caught.value.__notes__)
    shards = list(staging[0].glob("*.safetensors"))
    assert shards
    for shard in shards:
        with safe_open(shard, framework="pt") as reader:
            assert list(reader.keys())


def test_autoround_preserves_checkpoint_after_save_failure(tmp_path):
    run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "failure", str(tmp_path), "2"],
        capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    {"produce": _produce, "reload": _reload, "failure": _produce_failure}[sys.argv[1]](Path(sys.argv[2]), int(sys.argv[3]))
