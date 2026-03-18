# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_example_glm_contains_calibration_generation_helper_and_flags():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "examples" / "quantization" / "examples" / "example_glm_4_7_flash.py"
    content = script_path.read_text(encoding="utf-8")

    assert "def build_calibration_generation_config(" in content
    assert "--calib-gen-enabled" in content
    assert "--calib-gen-provider" in content
    assert "calibration_generation=build_calibration_generation_config(" in content
    assert "if not args.calib_gen_enabled:" in content
    assert "calib_dataset = None" in content
