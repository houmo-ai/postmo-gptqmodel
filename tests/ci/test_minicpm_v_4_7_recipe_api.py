"""CPU contract for the MiniCPM-V-4.7 recipe."""

from gptqmodel.recipes.minicpm_v_4_7 import (
    quantize_minicpm_v_4_7_autoround,
    quantize_minicpm_v_4_7_gptq,
)


def test_minicpm_v_4_7_gptq_dry_run_keeps_dense_contract(tmp_path):
    result = quantize_minicpm_v_4_7_gptq(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        calibration_jsonl=str(tmp_path / "calibration.jsonl"),
        dry_run=True,
    )

    assert result.method == "gptq"
    assert result.artifact_format == "float_causal_bin"
    assert result.model_family == "minicpm_v_4_7"
    assert result.topology == "dense"
    assert result.precision["backbone"] == "qwen3_5"
    assert result.precision["bits"] == 4
    assert result.precision["group_size"] == 64
    assert result.calibration["nsamples"] == 128
    assert result.provenance["dry_run"] is True
    assert not (tmp_path / "out").exists()


def test_minicpm_v_4_7_autoround_dry_run_keeps_moe_contract(tmp_path):
    result = quantize_minicpm_v_4_7_autoround(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
    )

    assert result.method == "autoround"
    assert result.topology == "moe"
    assert result.model_family == "minicpm_v_4_7"
    assert result.provenance["dry_run"] is True
    assert not (tmp_path / "out").exists()
