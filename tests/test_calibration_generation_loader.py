# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from gptqmodel.quantization.calibration.generation.jsonl_loader import load_calibration_jsonl


def test_load_calibration_jsonl_success(tmp_path: Path):
    p = tmp_path / "calib.jsonl"
    p.write_text(
        json.dumps({"text": "hello"}) + "\n" + json.dumps({"text": "world"}) + "\n",
        encoding="utf-8",
    )
    rows = load_calibration_jsonl(str(p))
    assert rows == [{"text": "hello"}, {"text": "world"}]


def test_load_calibration_jsonl_empty_should_fail(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_calibration_jsonl(str(p))