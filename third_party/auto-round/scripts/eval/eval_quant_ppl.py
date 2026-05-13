"""量化模型 PPL 评测脚本。

Qwen3.5 是 VLM 架构 (ForConditionalGeneration)，需要用原始架构类加载，
而非 AutoModelForCausalLM，否则 quantization_config 不会被正确识别。

用法:
  python eval_quant_ppl.py \
    --model weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128 \
    --device cuda:0
"""
import argparse
import os
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="量化模型 PPL 评测 (wikitext)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="wikitext")
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default=None,
                        help="显式指定 dtype (bfloat16/float16)，BF16 原始模型评测时使用")
    args = parser.parse_args()

    # Block gptqmodel C++ backends that require GLIBC_2.32 (system has 2.31)
    import auto_round.inference.backend as _bk
    for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
        del _bk.BackendInfos[k]

    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Detect model architecture to choose correct loading method
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])

    load_kwargs = dict(device_map=args.device, trust_remote_code=True)
    if args.dtype:
        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    if "Qwen3_5ForConditionalGeneration" in archs:
        # VLM model: must use specific class (AutoModelForCausalLM loses quantization_config)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, **load_kwargs,
        )
    else:
        # CausalLM model (e.g. rotated model saved as Qwen3_5ForCausalLM)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **load_kwargs,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.eval_bs,
        dtype="auto",
    )

    tasks = args.tasks.split(",")
    st = time.time()
    res = lm_eval.simple_evaluate(
        model=hflm,
        model_args=None,
        tasks=tasks,
        batch_size=args.eval_bs,
    )
    print(make_table(res))
    print(f"evaluation running time={int(time.time() - st)}s")


if __name__ == "__main__":
    main()
