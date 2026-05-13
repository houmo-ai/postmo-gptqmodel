import json
import sys
import types

from scripts_gemma4.eval import eval_ppl


def test_build_result_payload_contains_expected_fields():
    payload = eval_ppl.build_result_payload(
        model="/tmp/gemma4-e2b",
        seq_len=2048,
        stride=1024,
        device="cuda:0",
        dtype="bfloat16",
        ppl=12.34,
        mean_nll=2.51,
        n_tokens=4096,
        elapsed=99,
    )

    assert payload == {
        "model": "/tmp/gemma4-e2b",
        "seq_len": 2048,
        "stride": 1024,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "ppl": 12.34,
        "mean_nll": 2.51,
        "n_tokens": 4096,
        "elapsed_seconds": 99,
    }


def test_write_result_payload_persists_json(tmp_path):
    path = tmp_path / "ppl.json"
    payload = {"ppl": 1.23}

    eval_ppl.write_result_payload(str(path), payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_load_model_uses_auto_device_map_for_quantized_cuda_device(tmp_path, monkeypatch):
    model_dir = tmp_path / "gemma4-e2b-w4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "quantization_config": {"quant_method": "gptq"},
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    class _FakeGPTQModel:
        @staticmethod
        def load(model_path, device_map=None, backend=None, trust_remote_code=None):
            captured["model_path"] = model_path
            captured["device_map"] = device_map
            captured["backend"] = backend
            captured["trust_remote_code"] = trust_remote_code
            return types.SimpleNamespace(model=types.SimpleNamespace())

    fake_gptqmodel_module = types.SimpleNamespace(
        GPTQModel=_FakeGPTQModel,
        BACKEND=types.SimpleNamespace(TORCH="torch"),
    )
    fake_transformers_module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )

    monkeypatch.setattr(eval_ppl, "ensure_local_gptqmodel_path", lambda: None)
    monkeypatch.setattr(eval_ppl, "patch_gptqmodel_for_gemma4", lambda: True)
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_module)

    model, tokenizer = eval_ppl.load_model(str(model_dir), "cuda:0")

    assert model is not None
    assert tokenizer is not None
    assert captured["model_path"] == str(model_dir)
    assert captured["device_map"] == "auto"
    assert captured["backend"] == "torch"
