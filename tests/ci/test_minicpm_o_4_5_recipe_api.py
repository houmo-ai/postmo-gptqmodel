from __future__ import annotations

from pathlib import Path

import pytest
from gptqmodel.recipes.minicpm_o_4_5 import quantize_minicpm_o_4_5


def test_minicpm_o_4_5_recipe_defaults_are_gptq_w4_g64(tmp_path: Path):
    result = quantize_minicpm_o_4_5(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        calibration_jsonl=str(tmp_path / "calibration.jsonl"),
        dry_run=True,
    )

    assert result.method == "gptq"
    assert result.artifact_format == "gptqmodel_hf"
    assert result.model_family == "minicpm_o_4_5"
    assert result.precision == {
        "backbone": "qwen3",
        "bits": 4,
        "group_size": 64,
        "symmetric": True,
        "desc_act": False,
        "damp_percent": 0.01,
    }
    assert result.calibration["nsamples"] == 128
    assert result.calibration["seqlen"] == 1024
    assert result.provenance["backend"] == "direct_gptqmodel_api"
    assert result.provenance["offload_to_disk"] is True
    assert not (tmp_path / "out").exists()


def test_minicpm_o_4_5_recipe_offload_defaults_to_sibling_path(tmp_path: Path):
    result = quantize_minicpm_o_4_5(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out" / "gptq_llm"),
        dry_run=True,
    )

    assert result.provenance["offload_to_disk_path"] == str(
        (tmp_path / "out" / "gptq_llm_offload").resolve()
    )


def test_minicpm_o_4_5_recipe_disables_offload_when_requested(tmp_path: Path):
    result = quantize_minicpm_o_4_5(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        offload_to_disk=False,
        offload_to_disk_path=str(tmp_path / "custom"),
        dry_run=True,
    )

    assert result.provenance["offload_to_disk"] is False
    assert result.provenance["offload_to_disk_path"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "autoround"}, "method must be 'gptq'"),
        ({"artifact_format": "hf"}, "artifact_format must be 'gptqmodel_hf'"),
        ({"bits": 0}, "bits must be positive"),
        ({"group_size": 0}, "group_size must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"nsamples": 0}, "nsamples must be positive"),
        ({"seqlen": 0}, "seqlen must be positive"),
    ],
)
def test_minicpm_o_4_5_recipe_rejects_invalid_args(
    tmp_path: Path,
    kwargs: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        quantize_minicpm_o_4_5(
            model_dir=str(tmp_path / "model"),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            **kwargs,
        )
