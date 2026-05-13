"""Gemma4 量化模型 Benchmark 评测脚本 (lm_eval)。

支持 MMLU Pro, GPQA, ARC, HellaSwag, Winogrande 等 lm_eval 内置任务。
关键特性:
  - apply_chat_template: 对 instruct 模型使用 chat 格式（默认开启）
  - enable_thinking:     控制 Gemma4 的 thinking channel（默认关闭）
  - fewshot_as_multiturn: few-shot 示例作为多轮对话（默认开启）

用法:
  # BF16 原始模型 - MMLU Pro (推荐配置)
  CUDA_VISIBLE_DEVICES=2 python scripts_gemma4/eval/eval_benchmark.py \
    --model weights/gemma-4-31B-it --dtype bfloat16 --device cuda:0 \
    --tasks mmlu_pro --apply_chat_template --max_gen_toks 4096 \
    --output output/logs/eval-bf16-mmlu-pro-v2.json

  # Mode1 量化模型
  CUDA_VISIBLE_DEVICES=3 python scripts_gemma4/eval/eval_benchmark.py \
    --model output/gemma4-mode1 --device cuda:0 \
    --tasks mmlu_pro --apply_chat_template --max_gen_toks 4096 \
    --output output/logs/eval-mode1-mmlu-pro-v2.json

  # 不使用 chat template (raw completion mode, 不推荐用于 instruct 模型)
  python scripts_gemma4/eval/eval_benchmark.py \
    --model weights/gemma-4-31B-it --dtype bfloat16 --device cuda:0 \
    --tasks arc_challenge --no_chat_template
"""
import argparse
import json
import os
import time

import torch

from scripts_gemma4.eval.common import ensure_local_gptqmodel_path, patch_gptqmodel_for_gemma4

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path, device, dtype_str=None):
    """Load model with correct architecture class."""
    try:
        import auto_round.inference.backend as _bk
        for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
            del _bk.BackendInfos[k]
    except Exception:
        pass

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    is_quantized = "quantization_config" in cfg

    load_kwargs = dict(device_map=device, trust_remote_code=True)
    if dtype_str:
        load_kwargs["torch_dtype"] = getattr(torch, dtype_str)

    model = None

    if is_quantized:
        try:
            ensure_local_gptqmodel_path()
            patch_gptqmodel_for_gemma4()
            from gptqmodel import GPTQModel, BACKEND
            gptq_model = GPTQModel.load(
                model_path, device_map=device,
                backend=BACKEND.TORCH, trust_remote_code=True,
            )
            model = gptq_model.model
            print(f"Loaded via GPTQModel (TORCH backend): {type(model).__name__}")
        except Exception as e:
            print(f"GPTQModel.load failed: {e}, falling back...")

    if model is None:
        if "Gemma4ForConditionalGeneration" in archs:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        print(f"Loaded: {type(model).__name__}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Gemma4 Benchmark 评测 (lm_eval)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="gpqa_diamond_zeroshot",
                        help="逗号分隔的任务列表")
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="覆盖默认 few-shot 数量")
    parser.add_argument("--limit", type=float, default=None,
                        help="每个子任务最多评测的样本比例(0~1)或数量(>=1)，例如 0.1 表示 10%%")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    parser.add_argument("--max_gen_toks", type=int, default=None,
                        help="覆盖生成任务的最大 token 数（默认由任务定义，mmlu_pro=2048）。"
                             "CoT 任务推荐 4096 以确保完整推理链。")
    # Chat template 相关参数 (对 instruct 模型至关重要)
    parser.add_argument("--apply_chat_template", action="store_true", default=False,
                        help="使用 tokenizer 的 chat template 格式化 prompt（instruct 模型必须开启）")
    parser.add_argument("--no_chat_template", action="store_true", default=False,
                        help="强制不使用 chat template（raw completion mode）")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="启用 Gemma4 thinking channel（默认关闭，避免 thinking tokens 占用生成长度）")
    parser.add_argument("--system_instruction", type=str, default=None,
                        help="系统指令（可选，仅在 apply_chat_template 时生效）")
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=True,
                        help="few-shot 示例作为多轮对话（默认开启，仅在 apply_chat_template 时生效）")
    parser.add_argument("--no_fewshot_multiturn", action="store_true", default=False,
                        help="禁用 fewshot_as_multiturn")
    parser.add_argument("--log_samples", action="store_true", default=False,
                        help="在 JSON 输出中保存每个样本的详细结果（用于调试）")
    args = parser.parse_args()

    # Resolve chat template flag
    if args.no_chat_template:
        args.apply_chat_template = False

    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: {args.device}")
    print(f"apply_chat_template: {args.apply_chat_template}")
    print(f"enable_thinking: {args.enable_thinking}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")
    if args.max_gen_toks is not None:
        print(f"max_gen_toks: {args.max_gen_toks}")

    model, tokenizer = load_model(args.model, args.device, args.dtype)

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    # For VLM models like Gemma4, top-level config lacks max_position_embeddings,
    # causing HFLM to default to 2048. Read from text_config instead.
    max_len = getattr(getattr(model.config, "text_config", None),
                      "max_position_embeddings", None)

    hflm_kwargs = dict(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.eval_bs,
        dtype="auto",
        max_length=max_len,
    )

    # enable_thinking=False avoids Gemma4's thinking channel consuming tokens.
    # The model will output CoT reasoning directly in the response channel.
    if hasattr(model.config, "text_config"):  # VLM model (Gemma4)
        hflm_kwargs["enable_thinking"] = args.enable_thinking

    hflm = HFLM(**hflm_kwargs)

    tasks = [t.strip() for t in args.tasks.split(",")]
    fewshot_multiturn = args.fewshot_as_multiturn and not args.no_fewshot_multiturn
    eval_kwargs = dict(
        model=hflm,
        model_args=None,
        tasks=tasks,
        batch_size=args.eval_bs,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=fewshot_multiturn if args.apply_chat_template else False,
        log_samples=args.log_samples,
    )
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit
    if args.system_instruction:
        eval_kwargs["system_instruction"] = args.system_instruction
    if args.max_gen_toks is not None:
        eval_kwargs["gen_kwargs"] = f"max_gen_toks={args.max_gen_toks}"

    st = time.time()
    res = lm_eval.simple_evaluate(**eval_kwargs)
    table_str = make_table(res)
    print(table_str)
    elapsed = int(time.time() - st)
    print(f"\nEvaluation time: {elapsed}s")

    # Print summary
    print("\n" + "=" * 50)
    for task_name, task_res in res["results"].items():
        for metric, value in task_res.items():
            if metric.endswith(",none") and "stderr" not in metric:
                metric_name = metric.replace(",none", "")
                print(f"{task_name}/{metric_name}: {value}")
    print("=" * 50)

    # Save results to JSON if requested
    if args.output:
        import copy
        save_data = copy.deepcopy(res)
        if not args.log_samples:
            save_data.pop("samples", None)
        save_data["_meta"] = {
            "model": args.model,
            "tasks": args.tasks,
            "device": args.device,
            "limit": args.limit,
            "apply_chat_template": args.apply_chat_template,
            "enable_thinking": args.enable_thinking,
            "max_gen_toks": args.max_gen_toks,
            "elapsed_seconds": elapsed,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
