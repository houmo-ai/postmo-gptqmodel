import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts_qwen35moe" / "chat_cli.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("scripts_qwen35moe.chat_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_model_and_tokenizer_uses_gptqmodel_for_quantized_checkpoint(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": {"bits": 4}}), encoding="utf-8")

    module = _load_module()
    fake_model = object()
    load_calls = []
    tokenizer_calls = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        unk_token = "<unk>"

        @classmethod
        def from_pretrained(cls, model_path, trust_remote_code=False):
            tokenizer_calls.append((model_path, trust_remote_code))
            return cls()

    class FakeGPTQModel:
        @staticmethod
        def load(model_path, **kwargs):
            load_calls.append((model_path, kwargs))
            return SimpleNamespace(model=fake_model)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            AutoModelForImageTextToText=object(),
            AutoModelForCausalLM=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gptqmodel",
        SimpleNamespace(GPTQModel=FakeGPTQModel, BACKEND=SimpleNamespace(TORCH="torch")),
    )

    model, tokenizer = module.load_model_and_tokenizer(str(tmp_path), device_map="auto", dtype_name="bfloat16")

    assert model is fake_model
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer_calls == [(str(tmp_path), True)]
    assert load_calls == [
        (
            str(tmp_path),
            {
                "device_map": "auto",
                "backend": "torch",
                "trust_remote_code": True,
            },
        )
    ]


def test_render_prompt_falls_back_when_tokenizer_rejects_enable_thinking():
    module = _load_module()
    calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            if "enable_thinking" in kwargs:
                raise TypeError("enable_thinking unsupported")
            return "rendered prompt"

    prompt = module.render_prompt(FakeTokenizer(), [{"role": "user", "content": "hello"}], enable_thinking=False)

    assert prompt == "rendered prompt"
    assert calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        {"tokenize": False, "add_generation_prompt": True},
    ]


def test_run_repl_clears_history_but_keeps_system_prompt():
    module = _load_module()
    seen_messages = []
    outputs = []
    user_inputs = iter(["hello", "/clear", "world", "/exit"])

    def fake_reply_fn(model, tokenizer, messages, **kwargs):
        seen_messages.append(list(messages))
        return f"reply-{len(seen_messages)}"

    module.run_repl(
        model=object(),
        tokenizer=object(),
        system_prompt="be concise",
        max_new_tokens=32,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        input_fn=lambda _: next(user_inputs),
        output_fn=lambda message="": outputs.append(str(message)),
        reply_fn=fake_reply_fn,
    )

    assert seen_messages == [
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "world"},
        ],
    ]
    assert any("History cleared." in line for line in outputs)
    assert any("reply-1" in line for line in outputs)
    assert any("reply-2" in line for line in outputs)
