#!/bin/bash
# Monitor evaluations and finalize when both complete
# Usage: nohup bash scripts_gemma4/eval/wait_and_finalize.sh &

LOG_BF16="output/logs/eval-bf16-mmlu-pro-v2.log"
LOG_MODE1="output/logs/eval-mode1-mmlu-pro-v2.log"
JSON_BF16="output/logs/eval-bf16-mmlu-pro-v2.json"
JSON_MODE1="output/logs/eval-mode1-mmlu-pro-v2.json"

echo "[$(date)] Waiting for MMLU Pro v2 evaluations to complete..."

while true; do
    bf16_done=false
    mode1_done=false
    
    # Check if JSON result files exist (written at end of eval)
    [ -f "$JSON_BF16" ] && bf16_done=true
    [ -f "$JSON_MODE1" ] && mode1_done=true
    
    if $bf16_done && $mode1_done; then
        echo "[$(date)] Both evaluations complete!"
        break
    fi
    
    # Show progress
    if ! $bf16_done; then
        prog=$(grep "generate_until requests" "$LOG_BF16" 2>/dev/null | tail -1 | grep -oP '\d+/\d+' | tail -1)
        echo "[$(date)] BF16: $prog (waiting...)"
    else
        echo "[$(date)] BF16: DONE"
    fi
    if ! $mode1_done; then
        prog=$(grep "generate_until requests" "$LOG_MODE1" 2>/dev/null | tail -1 | grep -oP '\d+/\d+' | tail -1)
        echo "[$(date)] Mode1: $prog (waiting...)"
    else
        echo "[$(date)] Mode1: DONE"
    fi
    
    sleep 600  # check every 10 minutes
done

echo ""
echo "[$(date)] Running finalization..."
cd /data01/home/yujy/work/auto-round
/data01/home/yujy/miniconda3/envs/gemma4/bin/python scripts_gemma4/eval/finalize_mmlu_pro_v2.py
echo "[$(date)] Done!"
