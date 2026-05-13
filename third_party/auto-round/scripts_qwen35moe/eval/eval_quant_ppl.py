"""量化模型 PPL 评测脚本 (Qwen3.5-MoE)。

Qwen3.5-MoE 是 VLM 架构 (Qwen3_5MoeForConditionalGeneration)，需要用原始架构类加载，
而非 AutoModelForCausalLM，否则 quantization_config 不会被正确识别。

支持的架构:
  - Qwen3_5MoeForConditionalGeneration (MoE VLM)
  - Qwen3_5MoeForCausalLM (MoE CausalLM, e.g. rotated model)
  - Qwen3_5ForConditionalGeneration (dense VLM, backward compat)
  - AutoModelForCausalLM (fallback)

用法:
  # 量化模型评测
  python scripts_qwen35moe/eval/eval_quant_ppl.py \
    --model ./output/qwen35moe-mode3-rotate-full \
    --device cuda:0

  # BF16 原始模型评测
  python scripts_qwen35moe/eval/eval_quant_ppl.py \
    --model /data01/nfs_shared/Qwen3.5-35B-A3B \
    --device cuda:0 --dtype bfloat16
"""
import argparse
import os
import sys
import time

# Fix gptqmodel editable install path resolution
_gptq_root = os.path.expanduser("~/gptqmodel")
if os.path.isdir(_gptq_root) and _gptq_root not in sys.path:
    sys.path.insert(0, _gptq_root)

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force MoE fallback on GPUs without SM90
try:
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        if cc[0] < 9:
            import transformers.integrations.moe as _moe
            _moe._can_use_grouped_mm = lambda *a, **kw: False
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="量化模型 PPL 评测 (wikitext) - Qwen3.5-MoE")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="wikitext")
    parser.add_argument("--eval_bs", type=int, default=1,
                        help="lm-eval batch size。Qwen3.5-35B-A3B MoE 默认建议 1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default="auto",
                        help='模型加载 device_map，默认 auto。示例: auto、balanced、cuda:0')
    parser.add_argument("--dtype", type=str, default=None,
                        help="显式指定 dtype (bfloat16/float16)，BF16 原始模型评测时使用")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="lm-eval 单次前向的最大上下文长度。值越大越容易在 lm_head 处 OOM")
    parser.add_argument("--limit", type=float, default=None,
                        help="仅评测前 N 条样本；小于 1 时按比例截断，仅用于快速验证")
    args = parser.parse_args()

    # Block gptqmodel C++ backends that require GLIBC_2.32 (system has 2.31)
    try:
        import auto_round.inference.backend as _bk
        for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
            del _bk.BackendInfos[k]
    except Exception:
        pass

    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Detect model architecture to choose correct loading method
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])

    load_kwargs = dict(device_map=args.device_map, trust_remote_code=True)
    if args.dtype:
        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    model = None
    is_quantized = "quantization_config" in cfg

    # For quantized models, use gptqmodel's GPTQModel.load with TORCH backend
    # (Marlin backend fails on layers with out_features not divisible by 64)
    if is_quantized:
        try:
            from gptqmodel import GPTQModel, BACKEND
            gptq_model = GPTQModel.load(
                args.model, device_map=args.device_map,
                backend=BACKEND.TORCH, trust_remote_code=True,
            )
            model = gptq_model.model  # unwrap the inner HF model
            print(f"Loaded via GPTQModel (TORCH backend): {type(model).__name__}")
        except Exception as e:
            print(f"GPTQModel.load failed: {e}, trying other methods...")
    # breakpoint()
    # Try MoE VLM (for non-quantized models)
    if model is None and "Qwen3_5MoeForConditionalGeneration" in archs:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
            print(f"Loaded as AutoModelForImageTextToText: {type(model).__name__}")
        except Exception as e:
            print(f"AutoModelForImageTextToText failed: {e}, trying AutoModelForCausalLM...")

    # Try dense VLM
    if model is None and "Qwen3_5ForConditionalGeneration" in archs:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
            model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
            print(f"Loaded as Qwen3_5ForConditionalGeneration")
        except Exception as e:
            print(f"Qwen3_5ForConditionalGeneration failed: {e}")

    # Fallback to CausalLM
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        print(f"Loaded as AutoModelForCausalLM: {type(model).__name__}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    vocab_size = None
    text_cfg = cfg.get("text_config", {}) if isinstance(cfg.get("text_config"), dict) else {}
    vocab_size = text_cfg.get("vocab_size", cfg.get("vocab_size"))
    if vocab_size and args.max_length:
        logits_gib = args.eval_bs * args.max_length * vocab_size * 2 / (1024 ** 3)
        print(
            f"Estimated logits memory per forward: ~{logits_gib:.2f} GiB "
            f"(bs={args.eval_bs}, max_length={args.max_length}, vocab={vocab_size}, dtype~fp16/bf16)"
        )
        print("Note: device_map 只能切分层，lm_head 产生的 [bs, seq, vocab] logits 仍会落在同一张卡上。")

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.eval_bs,
        max_length=args.max_length,
        truncation=bool(args.max_length),
        dtype="auto",
    )

    tasks = args.tasks.split(",")
    st = time.time()
    res = lm_eval.simple_evaluate(
        model=hflm,
        model_args=None,
        tasks=tasks,
        batch_size=args.eval_bs,
        limit=args.limit,
    )
    print(make_table(res))
    print(f"evaluation running time={int(time.time() - st)}s")


if __name__ == "__main__":
    main()
