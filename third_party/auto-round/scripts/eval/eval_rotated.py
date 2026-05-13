"""旋转模型评测脚本：对 Qwen3.5-9B 应用 Hadamard 旋转后评测 PPL。

参考 gptqmodel 的 Qwen3.5 旋转实现:
  1. fuse_layer_norms — 将 RMSNorm 融合进相邻 Linear
  2. rotate_model — 用正交矩阵旋转权重

用法:
  # 仅旋转 + PPL 评测
  python eval_rotated.py --model weights/Qwen3.5-9B --device cuda:0

  # 旋转 + 保存 (供后续量化使用)
  python eval_rotated.py --model weights/Qwen3.5-9B --device cuda:0 --save_dir weights/Qwen3.5-9B-rotated
"""
import argparse
import os
import sys
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Qwen3.5 layer/module naming constants
LAYERS_NODE = "model.layers"
LM_HEAD_NAME = "lm_head"
PRE_HEAD_NORM = "model.norm"


def apply_rotation(model, device):
    """Apply Hadamard rotation to Qwen3.5ForCausalLM model using gptqmodel functions."""
    sys.path.insert(0, "/data01/home/yujy/work/gptqmodel")
    from gptqmodel.quantization.rotation.rotation import (
        fuse_layer_norms_qwen3_5,
        rotate_model_qwen3_5,
    )

    print("Step 1: Fusing LayerNorms into adjacent Linear layers...")
    model = fuse_layer_norms_qwen3_5(
        model,
        pre_lm_head_norm_module_name=PRE_HEAD_NORM,
        layers_node=LAYERS_NODE,
        lm_head_name=LM_HEAD_NAME,
    )

    print("Step 2: Applying Hadamard rotation...")
    model, Q = rotate_model_qwen3_5(
        model,
        rotate_mode="hadamard",
        device=torch.device(device),
        lm_head_name=LM_HEAD_NAME,
        layers_node=LAYERS_NODE,
    )

    print("Rotation complete.")
    return model, Q


def evaluate_ppl(model, tokenizer, eval_bs, tasks):
    """Evaluate PPL using lm_eval."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_bs, dtype="auto")
    st = time.time()
    res = lm_eval.simple_evaluate(model=hflm, model_args=None, tasks=tasks, batch_size=eval_bs)
    print(make_table(res))
    print(f"evaluation running time={int(time.time() - st)}s")
    return res


def main():
    parser = argparse.ArgumentParser(description="旋转模型 PPL 评测")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="wikitext")
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="保存旋转后模型的路径 (可选)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print(f"Model loaded: {type(model).__name__}")

    # Apply rotation on CPU (float64 operations), then move to GPU
    model, Q = apply_rotation(model, args.device)
    model = model.to(args.device)

    # Optionally save rotated model
    if args.save_dir:
        print(f"Saving rotated model to {args.save_dir}...")
        model_save = model.to("cpu")
        model_save.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        # Save Q matrix for reproducibility
        torch.save(Q, os.path.join(args.save_dir, "rotation_Q.pt"))
        print(f"Saved. Q matrix shape: {Q.shape}")
        model = model_save.to(args.device)
        del model_save

    # Evaluate PPL
    print("\nEvaluating PPL on rotated model...")
    evaluate_ppl(model, tokenizer, args.eval_bs, args.tasks.split(","))


if __name__ == "__main__":
    main()
