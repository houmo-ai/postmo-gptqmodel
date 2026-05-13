import json

from scripts_gemma4.eval import summarize_evalscope


def test_build_summary_formats_markdown_and_deltas(tmp_path):
    bf16_dir = tmp_path / "bf16"
    w4_dir = tmp_path / "w4"
    bf16_dir.mkdir()
    w4_dir.mkdir()

    bf16_payload = {
        "mmlu_pro": {"status": "ok", "primary_metric": "AverageAccuracy", "metrics": {"AverageAccuracy": 0.55}},
        "ceval": {"status": "ok", "primary_metric": "AverageAccuracy", "metrics": {"AverageAccuracy": 0.61}},
        "humaneval": {"status": "ok", "primary_metric": "pass@1", "metrics": {"pass@1": 0.29}},
    }
    w4_payload = {
        "mmlu_pro": {"status": "ok", "primary_metric": "AverageAccuracy", "metrics": {"AverageAccuracy": 0.52}},
        "ceval": {"status": "ok", "primary_metric": "AverageAccuracy", "metrics": {"AverageAccuracy": 0.60}},
        "humaneval": {"status": "blocked", "reason": "sandbox unavailable"},
    }

    (bf16_dir / "result_normalized.json").write_text(json.dumps(bf16_payload), encoding="utf-8")
    (w4_dir / "result_normalized.json").write_text(json.dumps(w4_payload), encoding="utf-8")

    summary, markdown = summarize_evalscope.build_summary(str(tmp_path), baseline="bf16")

    assert summary["bf16"]["mmlu_pro"]["delta_vs_baseline"] == "0.0000"
    assert summary["w4"]["mmlu_pro"]["delta_vs_baseline"] == "-0.0300"
    assert summary["w4"]["humaneval"]["metric"] == "blocked"
    assert "| w4 | humaneval | blocked | sandbox unavailable | N/A |" in markdown


def test_extract_score_handles_blocked_entries():
    metric, score = summarize_evalscope._extract_score({"status": "blocked", "reason": "sandbox unavailable"})

    assert metric == "blocked"
    assert score == "sandbox unavailable"
