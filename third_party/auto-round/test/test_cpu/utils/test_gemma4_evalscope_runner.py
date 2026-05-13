import argparse
import json
import sys
import types

from scripts_gemma4.eval import common as eval_common
from scripts_gemma4.eval import evalscope_eval


def test_build_task_config_uses_work_dir_for_cache_resume():
    args = argparse.Namespace(
        work_dir="/tmp/gemma4-evalscope",
        eval_batch_size=1,
        max_tokens=512,
        limit=None,
        debug=False,
        use_cache=True,
    )

    task_cfg = evalscope_eval._build_task_config(
        args=args,
        model_api=object(),
        model_id="gemma4-e2b",
        resolved_datasets=["mmlu_pro"],
    )

    assert task_cfg.use_cache == args.work_dir
    assert task_cfg.no_timestamp is True


def test_parse_dataset_subset_overrides_supports_ceval_and_mmlu_pro():
    overrides = evalscope_eval._parse_dataset_subset_overrides(
        [
            "ceval=advanced_mathematics",
            "mmlu_pro=biology",
        ]
    )

    assert overrides == {
        "ceval": ["advanced_mathematics"],
        "mmlu_pro": ["biology"],
    }


def test_build_model_api_kwargs_disables_thinking_by_default():
    args = argparse.Namespace(
        model="/tmp/model",
        device_map="auto",
        dtype="bfloat16",
        max_tokens=512,
        enable_thinking=False,
    )

    kwargs = evalscope_eval._build_model_api_kwargs(args, "gemma4-e2b")

    assert kwargs["enable_thinking"] is False


def test_normalize_results_marks_blocked_humaneval():
    normalized = evalscope_eval._normalize_results(
        raw_results={},
        requested_datasets=["mmlu_pro", "humaneval"],
        blocked={"humaneval": "sandbox unavailable"},
    )

    assert normalized["mmlu_pro"]["status"] == "ok"
    assert normalized["humaneval"] == {
        "dataset": "humaneval",
        "status": "blocked",
        "reason": "sandbox unavailable",
    }


def test_load_report_payloads_reads_requested_evalscope_outputs(tmp_path):
    model_id = "gemma4-e2b-w4"
    report_dir = tmp_path / "reports" / model_id
    report_dir.mkdir(parents=True)
    payload = {"score": 0.42, "metrics": [{"name": "pass@1", "score": 0.42}]}
    (report_dir / "humaneval.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = evalscope_eval._load_report_payloads(
        work_dir=str(tmp_path),
        model_id=model_id,
        requested_datasets=["humaneval"],
    )

    assert loaded == {"humaneval": payload}


def test_ensure_local_gptqmodel_path_prepends_existing_checkout(tmp_path, monkeypatch):
    fake_root = tmp_path / "gptqmodel"
    fake_root.mkdir()
    monkeypatch.setattr(eval_common.os.path, "isdir", lambda path: path == str(fake_root))
    monkeypatch.setattr(eval_common.os.path, "expanduser", lambda path: str(fake_root))
    monkeypatch.setattr(eval_common.sys, "path", ["existing"])

    inserted = eval_common.ensure_local_gptqmodel_path()

    assert inserted == str(fake_root)
    assert eval_common.sys.path[0] == str(fake_root)


def test_patch_gptqmodel_for_gemma4_registers_runtime_alias(monkeypatch):
    fake_auto = types.SimpleNamespace(MODEL_MAP={}, SUPPORTED_MODELS=[])
    fake_definitions = types.SimpleNamespace(Gemma3ForConditionalGenerationGPTQ=type("Gemma3Stub", (), {}))
    fake_models_module = types.ModuleType("gptqmodel.models")
    fake_models_module.auto = fake_auto
    fake_gemma3_module = types.ModuleType("gptqmodel.models.definitions.gemma3")
    fake_gemma3_module.Gemma3ForConditionalGenerationGPTQ = fake_definitions.Gemma3ForConditionalGenerationGPTQ

    monkeypatch.setattr(eval_common, "ensure_local_gptqmodel_path", lambda root=None: "/tmp/fake-gptqmodel")
    monkeypatch.setitem(sys.modules, "gptqmodel.models", fake_models_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.models.definitions.gemma3", fake_gemma3_module)

    patched = eval_common.patch_gptqmodel_for_gemma4()

    assert patched is True
    assert fake_auto.MODEL_MAP["gemma4"] is fake_definitions.Gemma3ForConditionalGenerationGPTQ
    assert "gemma4" in fake_auto.SUPPORTED_MODELS


def test_extract_multiple_choice_answer_supports_mmlu_pro_a_to_j():
    assert evalscope_eval._extract_multiple_choice_answer("ANSWER: J") == "J"
    assert evalscope_eval._extract_multiple_choice_answer("G)") == "G"


def test_detects_mmlu_pro_prompt_as_multiple_choice():
    prompt = """Question:
What is correct?
Options:
A) 1
B) 2
C) 3
D) 4
E) 5
F) 6
G) 7
H) 8
I) 9
J) 10
ANSWER: [LETTER]
"""

    assert evalscope_eval._is_multiple_choice_answer_prompt(prompt) is True
    assert evalscope_eval._get_multiple_choice_option_letters(prompt) == tuple("ABCDEFGHIJ")


def test_extract_multiple_choice_answer_supports_ceval_answer_format():
    assert evalscope_eval._extract_multiple_choice_answer("答案：C") == "C"


def test_detects_ceval_prompt_as_multiple_choice():
    prompt = """以下是中国关于高级数学的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。

问题：测试题
选项：
A) 选项一
B) 选项二
C) 选项三
D) 选项四
"""

    assert evalscope_eval._is_multiple_choice_answer_prompt(prompt) is True
