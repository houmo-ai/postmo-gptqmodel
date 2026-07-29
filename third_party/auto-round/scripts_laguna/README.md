# Laguna S 2.1 AutoRound example

This example quantizes routed expert projections to W4G64 and all other
quantized linear layers to W8G64. Embeddings, normalization weights, and
`lm_head` remain BF16 by default; pass `--quant-lm-head` to quantize the head to
W8 as well.

AutoRound's predefined exclusions also remain BF16. For Laguna S 2.1 these
include the MoE routing gates and the attention `g_proj` layers whose shapes are
not divisible by 32. The routed expert `gate_proj`, `up_proj`, and `down_proj`
weights are still all covered by the W4 override.

The launcher maps physical GPUs 4, 5, 6, and 7 to logical AutoRound devices
0, 1, 2, and 3:

```bash
bash third_party/auto-round/scripts_laguna/run_laguna_s21_w4_experts_w8_rest.sh
```

Common overrides:

```bash
AR_ITERS=50 AR_NSAMPLES=32 AR_BATCH_SIZE=1 \
  bash third_party/auto-round/scripts_laguna/run_laguna_s21_w4_experts_w8_rest.sh \
  /path/to/Laguna-S-2.1 /path/to/output
```

Run a meta-device structure check without allocating the 235 GB BF16 weights:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH="$PWD/third_party/auto-round:$PWD" \
/data01/home/xuzk/anaconda3/envs/gptqmodel/bin/python \
  third_party/auto-round/scripts_laguna/quantize_laguna.py \
  --model /data01/datasets/Laguna-S-2.1 --inspect-only
```

The default `auto_round` output format is recommended for this mixed-bit
configuration. Loading the result through Transformers also requires the local
AutoRound package on `PYTHONPATH` until the Laguna layout fix is available in an
installed AutoRound release.