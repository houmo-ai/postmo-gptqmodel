import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.input_cache import InputCache
from gptqmodel.models.definitions.gemma4 import (
    Gemma4PerLayerInputMixin,
    Gemma4UnifiedForConditionalGenerationGPTQ,
    _install_rotary_fix_hooks,
)
from gptqmodel.recipes.gemma4 import (
    _copy_hf_auxiliary_files,
    _infer_variant,
    _patch_gemma4_quantization_config,
    _restore_source_config_fields,
    quantize_gemma4,
)
from gptqmodel.utils.looper_helpers import forward_batch_worker


def test_gemma4_unified_has_explicit_quant_model_definition(tmp_path):
    from gptqmodel.models.auto import MODEL_MAP, check_and_get_model_definition

    model_dir = tmp_path / "gemma-4-12B-it"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "num_global_key_value_heads": 1,
                    "max_position_embeddings": 128,
                    "sliding_window": 32,
                    "layer_types": ["sliding_attention"],
                    "vocab_size": 128,
                },
                "vision_config": {"model_type": "gemma4_unified_vision"},
                "audio_config": {"model_type": "gemma4_unified_audio"},
            }
        ),
        encoding="utf-8",
    )

    assert MODEL_MAP["gemma4_unified"] is Gemma4UnifiedForConditionalGenerationGPTQ
    assert (
        check_and_get_model_definition(str(model_dir))
        is Gemma4UnifiedForConditionalGenerationGPTQ
    )
    assert Gemma4UnifiedForConditionalGenerationGPTQ.lm_head_module == "lm_head"
    assert (
        Gemma4UnifiedForConditionalGenerationGPTQ.pre_lm_head_norm_module
        == "model.language_model.norm"
    )
    assert Gemma4UnifiedForConditionalGenerationGPTQ.layer_modules_strict is True
    blocks = Gemma4UnifiedForConditionalGenerationGPTQ.build_layer_modules(
        Gemma4UnifiedForConditionalGenerationGPTQ.module_tree
    )
    for module_name in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        assert any(module_name in block for block in blocks)


def test_gemma4_unified_recomputes_rotary_width_for_each_layer_type():
    class FakeRotary(torch.nn.Module):
        def forward(self, hidden_states, position_ids, layer_type):
            dim = 512 if layer_type == "full_attention" else 256
            shape = (*position_ids.shape, dim)
            return torch.zeros(shape), torch.ones(shape)

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states, *, position_embeddings=None, position_ids=None):
            return position_embeddings[0]

    text_model = torch.nn.Module()
    text_model.rotary_emb = FakeRotary()
    text_model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
    core_model = torch.nn.Module()
    core_model.language_model = text_model
    model = torch.nn.Module()
    model.model = core_model
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4,
            layer_types=["sliding_attention", "full_attention"],
        )
    )

    _install_rotary_fix_hooks(model, config)
    hidden_states = torch.zeros(1, 3, 4)
    position_ids = torch.arange(3).unsqueeze(0)
    wrong_position_embeddings = (torch.zeros(1, 3, 256), torch.ones(1, 3, 256))

    actual = text_model.layers[1](
        hidden_states,
        position_embeddings=wrong_position_embeddings,
        position_ids=position_ids,
    )

    assert actual.shape[-1] == 512


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


def test_gemma4_unified_recipe_reports_12b_variant(tmp_path):
    model_dir = tmp_path / "gemma-4-12B-it"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "text_config": {"enable_moe_block": False},
                "audio_config": {"model_type": "gemma4_unified_audio"},
            }
        ),
        encoding="utf-8",
    )

    result = quantize_gemma4(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        check_quant_text_demo=False,
        dry_run=True,
    )

    assert result.variant == "12b_unified"
    assert result.topology == "dense"


def test_gemma4_recipe_treats_non_mapping_config_as_unknown(tmp_path):
    model_dir = tmp_path / "malformed-config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("[]", encoding="utf-8")

    assert _infer_variant(model_dir=str(model_dir)) == "unknown"


def test_autoround_registers_gemma4_unified_text_only_contract():
    from auto_round.special_model_handler import (
        SPECIAL_MULTIMODAL_BLOCK,
        SPECIAL_SHARED_CACHE_KEYS,
        SUPPORT_ONLY_TEXT_MODELS,
    )

    assert "gemma4_unified" in SUPPORT_ONLY_TEXT_MODELS
    assert (
        SPECIAL_MULTIMODAL_BLOCK["gemma4_unified"]
        is SPECIAL_MULTIMODAL_BLOCK["gemma4"]
    )
    assert (
        SPECIAL_SHARED_CACHE_KEYS["Gemma4UnifiedForConditionalGeneration"]
        == SPECIAL_SHARED_CACHE_KEYS["Gemma4ForConditionalGeneration"]
    )


def test_autoround_patch_supplies_unified_shared_kv_dictionary():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "auto-round"
        / "scripts_gemma4"
        / "quantize.py"
    )
    spec = importlib.util.spec_from_file_location("gemma4_autoround_quantize", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch_position_embeddings = module.patch_position_embeddings

    class FakeRotary(torch.nn.Module):
        def forward(self, hidden_states, position_ids, layer_type):
            dim = 8 if layer_type == "full_attention" else 4
            shape = (*position_ids.shape, dim)
            return torch.zeros(shape), torch.ones(shape)

    class FakeLayer(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            *,
            shared_kv_states=None,
            position_embeddings=None,
            **kwargs,
        ):
            shared_kv_states["stored"] = hidden_states
            self.received_shared_kv_states = shared_kv_states
            return position_embeddings[0]

    text_model = torch.nn.Module()
    text_model.rotary_emb = FakeRotary()
    text_model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
    core_model = torch.nn.Module()
    core_model.language_model = text_model
    model = torch.nn.Module()
    model.model = core_model
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4,
            layer_types=["sliding_attention", "full_attention"],
        )
    )

    patch_position_embeddings(model, seqlen=3, device="cpu")
    output = text_model.layers[1](
        torch.zeros(1, 3, 4),
        position_ids=torch.arange(3).unsqueeze(0),
    )

    assert output.shape[-1] == 8
    assert "stored" in text_model.layers[1].received_shared_kv_states


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


def test_gemma4_recipe_restores_source_only_multimodal_config_fields(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    source_config = {
        "model_type": "gemma4_unified",
        "architectures": ["SourceArchitecture"],
        "quantization_config": {"bits": 8},
        "vision_config": {"model_patch_size": 48, "num_soft_tokens": 280},
        "audio_config": {"audio_samples_per_token": 640},
        "text_config": {
            "hidden_size": 3840,
            "dtype": None,
            "source_only_runtime_field": True,
        },
    }
    quant_config = {
        "model_type": "gemma4_unified",
        "architectures": ["QuantizedArchitecture"],
        "quantization_config": {"bits": 4, "group_size": 64},
        "vision_config": {"num_soft_tokens": 280, "dtype": "bfloat16"},
        "text_config": {"hidden_size": 3840, "dtype": "bfloat16"},
    }
    (src / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(quant_config), encoding="utf-8")

    _restore_source_config_fields(str(src), str(out))

    actual = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert actual["vision_config"]["model_patch_size"] == 48
    assert actual["audio_config"]["audio_samples_per_token"] == 640
    assert actual["vision_config"]["dtype"] == "bfloat16"
    assert actual["text_config"]["dtype"] == "bfloat16"
    assert actual["quantization_config"] == {"bits": 4, "group_size": 64}
    assert actual["architectures"] == ["QuantizedArchitecture"]
    assert "source_only_runtime_field" not in actual["text_config"]


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
