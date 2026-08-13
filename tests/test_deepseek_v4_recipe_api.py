from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.recipes.deepseek_v4_flash import (
    audit_deepseek_v4_flash_quantized_artifact,
    build_deepseek_v4_flash_dynamic,
    prepare_deepseek_v4_flash_bf16,
    quantize_deepseek_v4_flash,
    select_deepseek_v4_flash_checkpoint_keys,
    tokenize_deepseek_v4_flash_calibration,
)


def _write_tiny_packed_artifact(root: Path, *, invalid_scale: bool = False) -> None:
    root.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "n_routed_experts": 1,
        "n_shared_experts": 1,
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "q_lora_rank": 64,
        "num_attention_heads": 1,
        "head_dim": 64,
        "o_groups": 1,
        "o_lora_rank": 64,
    }
    dynamic = build_deepseek_v4_flash_dynamic()
    quant_config = {
        "bits": 8,
        "group_size": 64,
        "sym": True,
        "dynamic": dynamic,
    }
    config["quantization_config"] = quant_config
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "quantize_config.json").write_text(
        json.dumps(quant_config),
        encoding="utf-8",
    )

    base = "model.layers.0"
    prefixes = [
        f"{base}.self_attn.q_a_proj",
        f"{base}.self_attn.q_b_proj",
        f"{base}.self_attn.kv_proj",
        f"{base}.self_attn.o_b_proj",
        f"{base}.mlp.shared_experts.gate_proj",
        f"{base}.mlp.shared_experts.up_proj",
        f"{base}.mlp.shared_experts.down_proj",
        f"{base}.mlp.experts.0.gate_proj",
        f"{base}.mlp.experts.0.up_proj",
        f"{base}.mlp.experts.0.down_proj",
    ]
    tensors = {}
    for prefix in prefixes:
        bits = 4 if ".mlp.experts.0." in prefix else 8
        pack = 32 // bits
        tensors[f"{prefix}.qweight"] = torch.zeros(64 // pack, 64, dtype=torch.int32)
        tensors[f"{prefix}.qzeros"] = torch.zeros(1, 64 // pack, dtype=torch.int32)
        scale_rows = 2 if invalid_scale and prefix == prefixes[0] else 1
        tensors[f"{prefix}.scales"] = torch.ones(scale_rows, 64, dtype=torch.float16)
        tensors[f"{prefix}.g_idx"] = torch.arange(64, dtype=torch.int32)
    shard = "model.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )


def test_deepseek_v4_recipe_defaults_to_w8_w4_g64_without_offload(tmp_path: Path):
    result = quantize_deepseek_v4_flash(
        model_dir=str(tmp_path / "native"),
        prepared_model_dir=str(tmp_path / "bf16"),
        output_dir=str(tmp_path / "w4"),
        num_layers=6,
        dry_run=True,
    )

    assert result.method == "gptq"
    assert result.num_layers == 6
    assert result.offload_to_disk is False
    assert result.provenance["offload_path"] is None
    assert result.precision["routed_non_shared_experts"] == "W4"
    assert result.precision["shared_expert_and_other_quantized_linears"] == "W8"
    assert result.precision["group_size"] == 64
    assert result.provenance["auto_forward_data_parallel"] is False
    assert (
        result.provenance["mla_weight_form"]
        == "shared_latent_and_grouped_output_preserved"
    )
    assert result.provenance["auto_round"]["enabled"] is False
    assert not (tmp_path / "w4").exists()


def test_deepseek_v4_packed_artifact_audit_proves_physical_w4_w8_inventory(
    tmp_path: Path,
):
    artifact = tmp_path / "packed"
    _write_tiny_packed_artifact(artifact)

    audit = audit_deepseek_v4_flash_quantized_artifact(artifact)

    assert audit.num_layers == 1
    assert audit.quantized_module_count == 10
    assert audit.w4_module_count == 3
    assert audit.w8_module_count == 7
    assert audit.shard_count == 1


def test_deepseek_v4_packed_artifact_audit_rejects_wrong_tensor_shape(
    tmp_path: Path,
):
    artifact = tmp_path / "packed"
    _write_tiny_packed_artifact(artifact, invalid_scale=True)

    with pytest.raises(ValueError, match="invalid packed tensor"):
        audit_deepseek_v4_flash_quantized_artifact(artifact)


def test_deepseek_v4_dynamic_sets_only_routed_experts_w4_and_skips_trailing_layers():
    dynamic = build_deepseek_v4_flash_dynamic(
        num_layers=6,
        max_quant_layers=2,
    )

    assert dynamic[r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"] == {
        "bits": 4,
        "group_size": 64,
        "sym": True,
    }
    for layer_index in range(2, 6):
        assert dynamic[rf"-:.*\.layers\.{layer_index}\..*"] == {}


def test_deepseek_v4_dynamic_does_not_match_shared_expert():
    from gptqmodel.quantization.config import dynamic_get

    dynamic = build_deepseek_v4_flash_dynamic()

    # Layers 0-2 use tid2eid hash routing, but still execute the same physical
    # expert matrices and therefore must not escape the routed-expert W4 rule.
    for layer in range(3):
        routed = f"model.layers.{layer}.mlp.experts.17.gate_proj"
        shared = f"model.layers.{layer}.mlp.shared_experts.gate_proj"
        assert dynamic_get(dynamic, routed, "bits", 8) == 4
        assert dynamic_get(dynamic, shared, "bits", 8) == 8


def test_deepseek_v4_recipe_disk_offload_is_explicit_opt_in(tmp_path: Path):
    result = quantize_deepseek_v4_flash(
        model_dir=str(tmp_path / "native"),
        prepared_model_dir=str(tmp_path / "bf16"),
        output_dir=str(tmp_path / "out"),
        offload_to_disk=True,
        offload_path=str(tmp_path / "offload"),
        dry_run=True,
    )

    assert result.offload_to_disk is True
    assert result.provenance["offload_path"] == str((tmp_path / "offload").resolve())


def test_deepseek_v4_forward_data_parallel_is_explicit_opt_in(tmp_path: Path):
    result = quantize_deepseek_v4_flash(
        model_dir=str(tmp_path / "native"),
        prepared_model_dir=str(tmp_path / "bf16"),
        output_dir=str(tmp_path / "out"),
        auto_forward_data_parallel=True,
        dry_run=True,
    )

    assert result.provenance["auto_forward_data_parallel"] is True


def test_deepseek_v4_autoround_builds_in_tree_mixed_precision_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import gptqmodel.recipes.deepseek_v4_flash as recipe_module
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AutoRoundVersion, METHOD

    model_dir = tmp_path / "bf16"
    output_dir = tmp_path / "autoround"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 1,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    class _Model:
        def quantize(self, rows, batch_size):
            captured["rows"] = rows
            captured["batch_size"] = batch_size

        def save(self, path):
            captured["save_path"] = path

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2] for _ in texts]}

    def _load(path, *, quantize_config, **kwargs):
        captured["load_path"] = path
        captured["qcfg"] = quantize_config
        return _Model()

    monkeypatch.setattr(GPTQModel, "load", staticmethod(_load))
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _Tokenizer()
    )
    monkeypatch.setattr(
        recipe_module,
        "audit_deepseek_v4_flash_quantized_artifact",
        lambda _path: None,
    )

    result = quantize_deepseek_v4_flash(
        model_dir=str(model_dir),
        prepared_model_dir=str(model_dir),
        output_dir=str(output_dir),
        method="autoround",
        calibration_data=["calibration"],
        nsamples=1,
        seqlen=8,
        device="cpu",
        auto_forward_data_parallel=False,
        auto_round_version="v2",
        auto_round_iters=1,
    )

    qcfg = captured["qcfg"]
    assert result.method == "auto_round"
    assert qcfg.quant_method == METHOD.AUTO_ROUND
    assert qcfg.auto_round_version == AutoRoundVersion.V2
    assert qcfg.bits == 8
    assert qcfg.group_size == 64
    assert qcfg.sym is True
    assert qcfg.offload_to_disk is False
    assert (
        qcfg.dynamic_get("model.layers.0.mlp.experts.17.down_proj", "bits", qcfg.bits)
        == 4
    )
    assert (
        qcfg.dynamic_get(
            "model.layers.0.mlp.shared_experts.down_proj", "bits", qcfg.bits
        )
        == 8
    )
    assert captured["batch_size"] == 1
    assert captured["save_path"] == str(output_dir.resolve())


def test_deepseek_v4_key_selection_keeps_globals_and_drops_mtp():
    weight_map = {
        "embed.weight": "a.safetensors",
        "layers.0.attn.wkv.weight": "a.safetensors",
        "layers.0.attn.wkv.scale": "a.safetensors",
        "layers.1.attn.wkv.weight": "b.safetensors",
        "mtp.0.attn.wkv.weight": "c.safetensors",
        "head.weight": "d.safetensors",
    }

    assert select_deepseek_v4_flash_checkpoint_keys(
        weight_map,
        num_layers=1,
    ) == {
        "embed.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
        "head.weight",
    }


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="float8_e8m0fnu dtype is required",
)
def test_prepare_deepseek_v4_first_layer_streams_and_rewrites_config(tmp_path: Path):
    source = tmp_path / "native"
    output = tmp_path / "bf16-first1"
    source.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "compress_ratios": [0, 4, 128],
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 1,
        "dspark_target_layer_ids": [1],
        "expert_dtype": "fp4",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    fp8 = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 2.0).to(torch.float8_e8m0fnu)
    tensors = {
        "embed.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "layers.0.attn.wkv.weight": fp8,
        "layers.0.attn.wkv.scale": scale,
        "layers.1.attn.wkv.weight": fp8.clone(),
        "layers.1.attn.wkv.scale": scale.clone(),
        "mtp.0.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "head.weight": torch.ones((2, 2), dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, source / shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: shard for key in tensors},
            }
        ),
        encoding="utf-8",
    )

    result = prepare_deepseek_v4_flash_bf16(
        model_dir=str(source),
        output_dir=str(output),
        num_layers=1,
        device=None,
    )

    assert result.num_layers == 1
    assert result.shard_count == 1
    with safe_open(output / shard, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {
            "embed.weight",
            "layers.0.attn.wkv.weight",
            "head.weight",
        }
        torch.testing.assert_close(
            reader.get_tensor("layers.0.attn.wkv.weight"),
            torch.full((128, 128), 2.0, dtype=torch.bfloat16),
        )
    prepared_config = json.loads((output / "config.json").read_text())
    assert prepared_config["num_hidden_layers"] == 1
    assert prepared_config["compress_ratios"] == [0]
    assert prepared_config["num_hash_layers"] == 1
    assert prepared_config["num_nextn_predict_layers"] == 0
    assert prepared_config["dspark_target_layer_ids"] == []
    assert "quantization_config" not in prepared_config
    assert "expert_dtype" not in prepared_config
    assert (output / "tokenizer_config.json").is_file()


def test_deepseek_v4_calibration_honors_seqlen():
    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["max_length"] == 3
            rows = [list(range(min(len(text), 3))) for text in texts]
            return {"input_ids": rows}

    calibration = tokenize_deepseek_v4_flash_calibration(
        ["abcdef", "x"],
        tokenizer=_Tokenizer(),
        seqlen=3,
    )

    assert calibration == [
        {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1]},
        {"input_ids": [0], "attention_mask": [1]},
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "awq"}, "method must be 'gptq' or 'auto_round'"),
        ({"bits": 4}, "requires base W8"),
        ({"expert_bits": 8}, "routed experts W4"),
        ({"group_size": 128}, "group_size=64"),
        ({"sym": False}, "symmetric"),
    ],
)
def test_deepseek_v4_recipe_rejects_other_deployment_profiles(
    tmp_path: Path,
    kwargs: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        quantize_deepseek_v4_flash(
            model_dir=str(tmp_path / "native"),
            prepared_model_dir=str(tmp_path / "bf16"),
            output_dir=str(tmp_path / "w4"),
            dry_run=True,
            **kwargs,
        )
