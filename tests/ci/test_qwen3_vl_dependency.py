# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import builtins
from pathlib import Path
import subprocess
import sys

import pytest

from gptqmodel.models.definitions import base_qwen3_vl
from gptqmodel.models.base import demo


def test_import_without_qwen_vl_utils():
    script = """
import builtins

real_import = builtins.__import__

def without_qwen_vl_utils(name, *args, **kwargs):
    if name == "qwen_vl_utils" or name.startswith("qwen_vl_utils."):
        raise ModuleNotFoundError("No module named 'qwen_vl_utils'", name="qwen_vl_utils")
    return real_import(name, *args, **kwargs)

builtins.__import__ = without_qwen_vl_utils

from gptqmodel import GPTQModel
from gptqmodel.recipes import quantize_qwen3_vl

assert callable(GPTQModel.load)
assert callable(quantize_qwen3_vl)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'qwen_vl_utils'", name="qwen_vl_utils"),
        ModuleNotFoundError("No module named 'av'", name="av"),
        ImportError("cannot import name 'process_vision_info'"),
    ],
)
def test_demo_optional_dependency_error(monkeypatch, error):
    real_import = builtins.__import__

    def fail_qwen_vl_import(name, *args, **kwargs):
        if name == "qwen_vl_utils":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qwen_vl_import)
    with pytest.raises(ImportError) as caught:
        demo(None, None)

    if isinstance(error, ModuleNotFoundError) and error.name == "qwen_vl_utils":
        assert "pip install qwen-vl-utils" in str(caught.value)
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error


def test_qwen3_vl_video_missing_dependency_has_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "qwen_vl_utils", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_vl_utils.vision_process", raising=False)

    real_import = builtins.__import__

    def fail_qwen_vl_import(name, *args, **kwargs):
        if name.startswith("qwen_vl_utils"):
            raise ImportError("No module named 'qwen_vl_utils'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qwen_vl_import)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "dummy.mp4"},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    with pytest.raises(ImportError, match="pip install qwen-vl-utils"):
        base_qwen3_vl.BaseQwen3VLGPTQ.process_vision_info(messages)
