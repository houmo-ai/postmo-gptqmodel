#!/bin/bash
# Collect MMLU Pro v2 results (with chat template fix)
# Run this after evaluations complete

echo "=== MMLU Pro v2 Results (with apply_chat_template fix) ==="
echo ""

for variant in bf16 mode1; do
    json_file="output/logs/eval-${variant}-mmlu-pro-v2.json"
    log_file="output/logs/eval-${variant}-mmlu-pro-v2.log"
    
    echo "--- ${variant^^} ---"
    if [ -f "$json_file" ]; then
        python3 -c "
import json
with open('$json_file') as f:
    data = json.load(f)
results = data.get('results', {})
for task, metrics in sorted(results.items()):
    em = metrics.get('exact_match,none', metrics.get('exact_match,flexible-extract', 'N/A'))
    em_se = metrics.get('exact_match_stderr,none', metrics.get('exact_match_stderr,flexible-extract', 'N/A'))
    if isinstance(em, (int, float)):
        em_pct = em * 100
        se_pct = em_se * 100 if isinstance(em_se, (int, float)) else 'N/A'
        print(f'  {task}: {em_pct:.2f}% (±{se_pct:.2f}%)' if isinstance(se_pct, float) else f'  {task}: {em_pct:.2f}%')
    else:
        print(f'  {task}: {em}')
"
    else
        echo "  [NOT READY] $json_file does not exist yet"
        if [ -f "$log_file" ]; then
            progress=$(grep "generate_until requests" "$log_file" | tail -1 | grep -oP '\d+/\d+' | tail -1)
            echo "  Progress: $progress"
        fi
    fi
    echo ""
done

echo "=== Comparison ==="
python3 -c "
import json, sys

results = {}
for variant in ['bf16', 'mode1']:
    path = f'output/logs/eval-{variant}-mmlu-pro-v2.json'
    try:
        with open(path) as f:
            data = json.load(f)
        r = data.get('results', {}).get('mmlu_pro', {})
        em = r.get('exact_match,none', r.get('exact_match,flexible-extract'))
        results[variant] = em * 100 if isinstance(em, (int, float)) else None
    except FileNotFoundError:
        results[variant] = None

bf16 = results.get('bf16')
mode1 = results.get('mode1')

if bf16 is not None:
    print(f'BF16:  {bf16:.2f}%  (official: 85.2%)')
if mode1 is not None:
    print(f'Mode1: {mode1:.2f}%')
if bf16 is not None and mode1 is not None:
    print(f'Delta: {mode1 - bf16:+.2f}pp')
if bf16 is None or mode1 is None:
    print('Some results not yet available.')
" 2>/dev/null || echo "Results not yet available."
