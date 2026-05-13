#!/usr/bin/env python3
"""
Finalize MMLU Pro v2 results: extract scores from JSON and update experiment report.
Run after both BF16 and Mode1 evaluations complete.

Usage:
    python scripts_gemma4/eval/finalize_mmlu_pro_v2.py
"""
import json
import re
import sys
from pathlib import Path


def load_score(json_path: str) -> dict | None:
    """Load MMLU Pro score from eval JSON output."""
    p = Path(json_path)
    if not p.exists():
        print(f"  [SKIP] {json_path} not found")
        return None
    with open(p) as f:
        data = json.load(f)
    results = data.get("results", {}).get("mmlu_pro", {})
    em = results.get("exact_match,none", results.get("exact_match,flexible-extract"))
    em_se = results.get("exact_match_stderr,none", results.get("exact_match_stderr,flexible-extract"))
    if em is None:
        print(f"  [WARN] No exact_match found in {json_path}")
        print(f"  Available keys: {list(results.keys())}")
        return None
    return {
        "score": em * 100,
        "stderr": em_se * 100 if em_se is not None else None,
    }


def update_report(bf16: dict, mode1: dict, report_path: str = "实验报告_Gemma4-31B_AutoRound.md"):
    """Update the experiment report with v2 results."""
    p = Path(report_path)
    if not p.exists():
        print(f"  [ERROR] Report not found: {report_path}")
        return False

    content = p.read_text()

    # Update v2 results table
    bf16_str = f"{bf16['score']:.2f}%" if bf16 else "_评测失败_"
    mode1_str = f"{mode1['score']:.2f}%" if mode1 else "_评测失败_"
    delta_str = f"{mode1['score'] - bf16['score']:+.2f}pp" if (bf16 and mode1) else "—"

    content = content.replace(
        "| BF16 | 50.12% | _评测中_ | 85.2% |",
        f"| BF16 | 50.12% | **{bf16_str}** | 85.2% |"
    )
    content = content.replace(
        "| Mode 1 | 44.75% | _评测中_ | — |",
        f"| Mode 1 | 44.75% | **{mode1_str}** | — |"
    )

    # Update quick comparison table
    if bf16:
        content = content.replace(
            "│ MMLU Pro (10%)v2 │ exact_match │ 评测中   │ 评测中   │ —        │",
            f"│ MMLU Pro (10%)v2 │ exact_match │ {bf16_str:8s} │ {mode1_str:8s} │ {delta_str:8s} │"
        )

    p.write_text(content)
    print(f"  [OK] Updated {report_path}")
    return True


def main():
    print("=== MMLU Pro v2 Results Finalization ===\n")

    bf16 = load_score("output/logs/eval-bf16-mmlu-pro-v2.json")
    mode1 = load_score("output/logs/eval-mode1-mmlu-pro-v2.json")

    print("\n--- Scores ---")
    if bf16:
        se = f" (±{bf16['stderr']:.2f}%)" if bf16['stderr'] else ""
        print(f"  BF16:   {bf16['score']:.2f}%{se}  (official: 85.2%)")
        gap = bf16['score'] - 85.2
        print(f"  Gap vs official: {gap:+.2f}pp")
    if mode1:
        se = f" (±{mode1['stderr']:.2f}%)" if mode1['stderr'] else ""
        print(f"  Mode1:  {mode1['score']:.2f}%{se}")
    if bf16 and mode1:
        delta = mode1['score'] - bf16['score']
        print(f"  Delta:  {delta:+.2f}pp")

    print("\n--- Updating Report ---")
    if bf16 or mode1:
        update_report(bf16, mode1)
    else:
        print("  [SKIP] No results to update")

    # Per-subject breakdown if available
    for variant, label in [("bf16", "BF16"), ("mode1", "Mode1")]:
        json_path = f"output/logs/eval-{variant}-mmlu-pro-v2.json"
        p = Path(json_path)
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        results = data.get("results", {})
        subjects = {k: v for k, v in results.items() if k.startswith("mmlu_pro_") and k != "mmlu_pro"}
        if subjects:
            print(f"\n--- {label} Per-Subject Top/Bottom 5 ---")
            scored = []
            for subj, metrics in subjects.items():
                em = metrics.get("exact_match,none", metrics.get("exact_match,flexible-extract"))
                if em is not None:
                    scored.append((subj.replace("mmlu_pro_", ""), em * 100))
            scored.sort(key=lambda x: x[1], reverse=True)
            print("  Top 5:")
            for name, score in scored[:5]:
                print(f"    {name}: {score:.1f}%")
            print("  Bottom 5:")
            for name, score in scored[-5:]:
                print(f"    {name}: {score:.1f}%")


if __name__ == "__main__":
    main()
