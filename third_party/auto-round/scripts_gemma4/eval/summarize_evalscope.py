import argparse
import json
import os
from pathlib import Path


DATASET_ORDER = ["mmlu_pro", "ceval", "humaneval"]


def _read_result_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_score(entry: dict) -> tuple[str, str]:
    status = entry.get("status", "unknown")
    if status != "ok":
        return status, entry.get("reason", status)
    primary_metric = entry.get("primary_metric")
    metrics = entry.get("metrics", {})
    if primary_metric and primary_metric in metrics:
        return primary_metric, str(metrics[primary_metric])
    if "score" in metrics:
        return "score", str(metrics["score"])
    return "score", "N/A"


def _format_delta(current: str, baseline: str) -> str:
    try:
        cur = float(current)
        base = float(baseline)
    except ValueError:
        return "N/A"
    return f"{cur - base:+.4f}"


def build_summary(root_dir: str, baseline: str | None) -> tuple[dict, str]:
    root = Path(root_dir)
    model_results = {}
    for result_path in sorted(root.glob("*/result_normalized.json")):
        model_name = result_path.parent.name
        model_results[model_name] = _read_result_file(str(result_path))

    lines = ["| model | dataset | metric | score | delta_vs_baseline |", "|---|---|---:|---:|---:|"]
    summary = {}
    baseline_result = model_results.get(baseline) if baseline else None

    for model_name, payload in model_results.items():
        summary[model_name] = {}
        for dataset in DATASET_ORDER:
            if dataset not in payload:
                continue
            metric_name, score = _extract_score(payload[dataset])
            baseline_score = "N/A"
            if baseline_result and dataset in baseline_result:
                _, baseline_score = _extract_score(baseline_result[dataset])
            delta = _format_delta(score, baseline_score) if model_name != baseline else "0.0000"
            summary[model_name][dataset] = {
                "metric": metric_name,
                "score": score,
                "delta_vs_baseline": delta,
                "raw": payload[dataset],
            }
            lines.append(f"| {model_name} | {dataset} | {metric_name} | {score} | {delta} |")

    return summary, "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize EvalScope normalized outputs")
    parser.add_argument("--root-dir", required=True, help="Directory containing per-model EvalScope outputs")
    parser.add_argument("--baseline", default=None, help="Model directory name used as delta baseline")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary, markdown = build_summary(args.root_dir, args.baseline)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
