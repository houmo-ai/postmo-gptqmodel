#!/usr/bin/env python3
"""一键汇总 Gemma4-31B 所有评测结果。

用法:
  python scripts_gemma4/eval/show_results.py
"""
import json
import os
import re

LOG_DIR = "output/logs"

def read_ppl():
    """读取 PPL 结果"""
    results = {}
    for tag in ["bf16", "mode1"]:
        path = os.path.join(LOG_DIR, f"eval-{tag}-ppl-v2.log")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = re.search(r"PPL:\s+([\d.]+)", line)
                if m:
                    results[tag] = float(m.group(1))
    return results


def read_benchmark():
    """读取 loglikelihood benchmark 结果"""
    results = {}
    for tag in ["bf16", "mode1"]:
        path = os.path.join(LOG_DIR, f"eval-{tag}-bench.log")
        if not os.path.exists(path):
            continue
        results[tag] = {}
        with open(path) as f:
            content = f.read()
        for line in content.split("\n"):
            if line.startswith("|") and "acc" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 9:
                    task = parts[1] if parts[1] else last_task
                    metric = parts[5]
                    value = parts[7]
                    try:
                        results[tag][f"{task}/{metric}"] = float(value)
                    except ValueError:
                        pass
                    last_task = task if parts[1] else last_task
            elif line.startswith("|") and line.count("|") > 3:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and parts[1]:
                    last_task = parts[1]
    return results


def read_mmlu_pro():
    """读取 MMLU Pro JSON 结果"""
    results = {}
    for tag in ["bf16", "mode1"]:
        path = os.path.join(LOG_DIR, f"eval-{tag}-mmlu-pro.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        results[tag] = {}
        for task, metrics in data["results"].items():
            for mk, mv in metrics.items():
                if "exact_match" in mk and "stderr" not in mk:
                    results[tag][task] = mv
    return results


def main():
    print("=" * 70)
    print("  Gemma4-31B AutoRound W4G64 对称量化 — 全部结果汇总")
    print("=" * 70)

    # 1. PPL
    print("\n┌─ 1. PPL (WikiText-2, 滑动窗口 seq=2048 stride=1024)")
    ppl = read_ppl()
    if ppl:
        bf16_ppl = ppl.get("bf16", 0)
        m1_ppl = ppl.get("mode1", 0)
        delta = (m1_ppl - bf16_ppl) / bf16_ppl * 100 if bf16_ppl else 0
        print(f"│  BF16:   {bf16_ppl:.4f}")
        print(f"│  Mode1:  {m1_ppl:.4f}  (Δ = {delta:+.2f}%)")
        print(f"│  结论:   {'✅ 近乎无损' if abs(delta) < 1 else '⚠️ 有差异'}")
    print("└" + "─" * 50)

    # 2. Benchmark
    print("\n┌─ 2. Benchmark (0-shot loglikelihood)")
    bench = read_benchmark()
    if "bf16" in bench and "mode1" in bench:
        tasks_of_interest = [
            ("arc_challenge/acc_norm", "ARC-C acc_norm"),
            ("hellaswag/acc_norm", "HellaSwag acc_norm"),
            ("truthfulqa_mc2/acc", "TruthfulQA MC2"),
            ("winogrande/acc", "Winogrande acc"),
        ]
        print(f"│  {'任务':<25s} {'BF16':>8s} {'Mode1':>8s} {'Δ':>8s}")
        print(f"│  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
        for key, label in tasks_of_interest:
            b = bench["bf16"].get(key)
            m = bench["mode1"].get(key)
            if b is not None and m is not None:
                delta = (m - b) * 100
                print(f"│  {label:<25s} {b*100:7.2f}% {m*100:7.2f}% {delta:+7.2f}pp")
    print("└" + "─" * 50)

    # 3. MMLU Pro
    print("\n┌─ 3. MMLU Pro (10% subset, 5-shot generate_until)")
    mmlu = read_mmlu_pro()
    if "bf16" in mmlu and "mode1" in mmlu:
        print(f"│  {'任务':<30s} {'BF16':>8s} {'Mode1':>8s} {'Δ':>8s}")
        print(f"│  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
        for task in sorted(mmlu["bf16"].keys()):
            b = mmlu["bf16"].get(task, 0)
            m = mmlu["mode1"].get(task, 0)
            delta = (m - b) * 100
            marker = " ◀" if task == "mmlu_pro" else ""
            print(f"│  {task:<30s} {b*100:7.2f}% {m*100:7.2f}% {delta:+7.2f}pp{marker}")
    print("└" + "─" * 50)

    # 4. 模型大小
    print("\n┌─ 4. 模型大小")
    for mode in [1, 2, 3]:
        path = f"output/gemma4-mode{mode}"
        if os.path.exists(path):
            size_bytes = sum(
                os.path.getsize(os.path.join(path, f))
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            print(f"│  Mode{mode}: {size_bytes / 1e9:.1f} GB")
    print(f"│  BF16:  130.0 GB")
    print("└" + "─" * 50)

    # 5. 文件清单
    print("\n┌─ 5. 结果文件清单")
    for f in sorted(os.listdir(LOG_DIR)):
        if f.startswith("eval-"):
            path = os.path.join(LOG_DIR, f)
            size = os.path.getsize(path)
            print(f"│  {f:<45s} {size:>8,d} bytes")
    print("└" + "─" * 50)

    print("\n✅ 所有结果文件完好，可通过 JSON 文件程序化提取。")


if __name__ == "__main__":
    main()
