from __future__ import annotations

import json
from pathlib import Path

import pytest

from gptqmodel.recipes.ling3_flash import (
    build_ling3_flash_dynamic,
    quantize_ling3_flash,
    tokenize_ling3_flash_calibration,
)


def test_ling_recipe_defaults_are_w8_w4_g64_without_offload(tmp_path: Path):
    result = quantize_ling3_flash(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        calibration_jsonl=str(tmp_path / "calibration.jsonl"),
        dry_run=True,
    )

    assert result.method == "gptq"
    assert result.artifact_format == "gptqmodel_hf"
    assert result.offload_to_disk is False
    assert result.provenance["offload_path"] is None
    assert result.provenance["wait_for_submodule_finalizers"] is True
    assert result.provenance["attention_backend"] == "eager"
    assert result.precision == {
        "routed_experts": "W4",
        "all_other_quantized_linears": "W8",
        "kda_decay_and_beta": "BF16",
        "mtp_layers": "BF16",
        "group_size": 64,
        "symmetric": True,
    }
    assert not (tmp_path / "out").exists()


def test_ling_recipe_excludes_mtp_blocks_from_gptq(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = quantize_ling3_flash(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
    )

    assert result.provenance["decoder_layers"] == 2
    assert result.provenance["mtp_layers"] == 1
    assert result.provenance["effective_max_quant_layers"] == 2
    assert result.provenance["dynamic"][r"-:.*\.layers\.2\..*"] == {}


def test_ling_recipe_disk_offload_is_explicit_opt_in(tmp_path: Path):
    result = quantize_ling3_flash(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        offload_to_disk=True,
        offload_path=str(tmp_path / "offload"),
        dry_run=True,
    )

    assert result.offload_to_disk is True
    assert result.provenance["offload_path"] == str((tmp_path / "offload").resolve())


def test_ling_dynamic_profile_only_overrides_routed_experts():
    dynamic = build_ling3_flash_dynamic(
        num_layers=4,
        max_quant_layers=2,
    )

    assert dynamic[
        r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
    ] == {"bits": 4, "group_size": 64, "sym": True}
    assert dynamic[r"-:.*\.layers\.2\..*"] == {}
    assert dynamic[r"-:.*\.layers\.3\..*"] == {}


def test_ling_calibration_honors_seqlen():
    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {
                "add_special_tokens": True,
                "truncation": True,
                "max_length": 4,
                "padding": False,
            }
            rows = [list(range(min(len(text), 4))) for text in texts]
            return {
                "input_ids": rows,
                "attention_mask": [[1] * len(row) for row in rows],
            }

    calibration = tokenize_ling3_flash_calibration(
        ["abcdefgh", "xy"],
        tokenizer=_Tokenizer(),
        seqlen=4,
    )

    assert calibration == [
        {"input_ids": [0, 1, 2, 3], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [0, 1], "attention_mask": [1, 1]},
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "autoround"}, "method must be 'gptq'"),
        ({"bits": 4}, "requires base W8"),
        ({"expert_bits": 8}, "routed experts W4"),
        ({"group_size": 128}, "group_size=64"),
        ({"sym": False}, "symmetric"),
    ],
)
def test_ling_recipe_rejects_non_deployment_profiles(
    tmp_path: Path,
    kwargs: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        quantize_ling3_flash(
            model_dir=str(tmp_path / "model"),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            **kwargs,
        )
