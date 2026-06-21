import json
from pathlib import Path

import pytest
import torch

from gptqmodel.looper.input_cache import InputCache
from gptqmodel.models.definitions.gemma4 import Gemma4PerLayerInputMixin
from gptqmodel.utils.looper_helpers import forward_batch_worker
from gptqmodel.recipes.gemma4 import _copy_hf_auxiliary_files, _patch_gemma4_quantization_config, quantize_gemma4


def test_gemma4_recipe_dry_run_builds_command(tmp_path):
    model_dir = tmp_path / "gemma-4-26B-A4B-it"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"enable_moe_block": True}}),
        encoding="utf-8",
    )

    result = quantize_gemma4(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        method="gptq",
        preset="full_multimodal",
        rotation=None,
        calibration_jsonl="calib.jsonl",
        calibration_text_key="text",
        nsamples=8,
        seqlen=2048,
        batch_size=1,
        offload_to_disk=False,
        trust_remote_code=True,
        check_quant_text_demo=False,
        dry_run=True,
    )

    command = result.provenance["command"]
    assert result.topology == "moe"
    assert result.variant == "26b_a4b"
    assert result.method == "gptq"
    assert result.output_dir == str((tmp_path / "out").resolve())
    assert "example_gemma4.py" in command[1]
    assert "--model" in command and str(model_dir) in command
    assert "--out" in command and str(tmp_path / "out") in command
    assert "--calibration-jsonl" in command and "calib.jsonl" in command
    assert "--no-offload-to-disk" in command
    assert "--skip-eval" in command


def test_gemma4_recipe_copies_processor_sidecars_without_overwriting_quant_config(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "processor_config.json").write_text("processor", encoding="utf-8")
    (src / "configuration.json").write_text("configuration", encoding="utf-8")
    (src / "config.json").write_text("source-config", encoding="utf-8")
    (src / "model.safetensors").write_text("weights", encoding="utf-8")
    (out / "config.json").write_text("quant-config", encoding="utf-8")

    _copy_hf_auxiliary_files(str(src), str(out))

    assert (out / "processor_config.json").read_text(encoding="utf-8") == "processor"
    assert (out / "configuration.json").read_text(encoding="utf-8") == "configuration"
    assert (out / "config.json").read_text(encoding="utf-8") == "quant-config"
    assert not (out / "model.safetensors").exists()


def test_gemma4_recipe_patches_hf_quant_reload_scope(tmp_path):
    out = tmp_path / "quant"
    out.mkdir()
    config = {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 64}}
    qconfig = {"quant_method": "gptq", "bits": 4, "group_size": 64}
    (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (out / "quantize_config.json").write_text(json.dumps(qconfig), encoding="utf-8")

    _patch_gemma4_quantization_config(str(out))

    patched_config = json.loads((out / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    patched_qconfig = json.loads((out / "quantize_config.json").read_text(encoding="utf-8"))
    for payload in (patched_config, patched_qconfig):
        assert payload["block_name_to_quantize"] == "model.language_model.layers"
        modules = payload["modules_in_block_to_quantize"]
        assert modules == [["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]]
        assert "per_layer_input_gate" not in modules[0]
        assert "per_layer_projection" not in modules[0]


def test_gemma4_recipe_rejects_rotation(tmp_path):
    model_dir = tmp_path / "gemma-4-31B-it"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="does not support rotation"):
        quantize_gemma4(
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            method="gptq",
            preset="full_multimodal",
            rotation="hadamard",
            dry_run=True,
        )


def test_gemma4_recipe_rejects_autoround_claim(tmp_path):
    model_dir = tmp_path / "gemma-4-E4B-it"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="method must be 'gptq'"):
        quantize_gemma4(
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            method="autoround",
            dry_run=True,
        )


def test_forward_worker_preserves_extra_positional_layer_inputs():
    class DummyProcessor:
        def _set_current_batch_index(self, batch_index):
            self.batch_index = batch_index

    class NeedsPerLayerInput(torch.nn.Module):
        def forward(self, hidden_states, per_layer_input=None, **kwargs):
            assert per_layer_input is not None
            assert kwargs["use_cache"] is False
            return hidden_states + per_layer_input

    hidden_states = torch.ones(1, 2, 3)
    per_layer_input = torch.full((1, 2, 3), 2.0)

    batch_idx, output, kv_next = forward_batch_worker(
        NeedsPerLayerInput(),
        DummyProcessor(),
        0,
        [hidden_states, per_layer_input],
        {},
        None,
        None,
        support_batch_quantize=False,
        is_lm_head_module=False,
        need_output=True,
        reuse_kv=False,
        prev_kv=None,
    )

    assert batch_idx == 0
    assert kv_next is None
    torch.testing.assert_close(output, torch.full((1, 2, 3), 3.0))


def test_gemma4_per_layer_input_hook_rebuilds_layer_positional_args():
    class FakeTextModel(torch.nn.Module):
        hidden_size_per_layer_input = 2

        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 3)

        def get_per_layer_inputs(self, input_ids, inputs_embeds):
            batch, seq = input_ids.shape
            layer0 = torch.zeros(batch, seq, 2)
            layer1 = torch.ones(batch, seq, 2)
            return torch.stack([layer0, layer1], dim=2)

        def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
            return per_layer_inputs + 2

    class FakeOuter:
        pass

    class FakeQModel(Gemma4PerLayerInputMixin):
        pass

    language_model = FakeTextModel()
    fake_model = FakeOuter()
    fake_model.model = FakeOuter()
    fake_model.model.language_model = language_model

    qmodel = FakeQModel()
    qmodel.model = fake_model
    example = {"input_ids": torch.tensor([[1, 2]])}
    layer_extra_inputs = qmodel.build_layer_extra_inputs_cache(example, torch.device("cpu"))
    input_cache = InputCache(
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        layer_extra_inputs=[[layer_extra_inputs[0]], [layer_extra_inputs[1]]],
    )

    hidden_states = torch.full((1, 2, 3), 7.0)
    prepared = qmodel.prepare_layer_inputs_for_quantize([[hidden_states]], input_cache, layer_index=1)

    assert prepared[0][0] is hidden_states
    torch.testing.assert_close(prepared[0][1], torch.full((1, 2, 2), 3.0))
