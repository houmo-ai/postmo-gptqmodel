# Laguna S 2.1 QuaRot + AutoRound example

The default pipeline first applies an offline Hadamard QuaRot transform and then
quantizes routed expert projections to W4G64 while keeping all other quantized
linear layers at W8G64. Embeddings, normalization weights, and `lm_head` remain
BF16 by default; pass `--quant-lm-head` to quantize the head to W8 as well.

QuaRot mutates the loaded BF16 model in place. The exact same model object is
then passed to AutoRound, so AutoRound calibrates and tunes the rotated weights
instead of reloading the original checkpoint. The transform covers the Laguna
router, attention `g_proj`, shared expert, packed routed experts, embeddings,
and `lm_head`; layers with either 48 or 72 attention heads are supported.

AutoRound's predefined exclusions also remain BF16. For Laguna S 2.1 these
include the MoE routing gates and the attention `g_proj` layers whose shapes are
not divisible by 32. The routed expert `gate_proj`, `up_proj`, and `down_proj`
weights are still all covered by the W4 override.

The launcher maps physical GPUs 4, 5, 6, and 7 to logical AutoRound devices
0, 1, 2, and 3. After export it loads the checkpoint and runs a real generation
check by default:

```bash
bash third_party/auto-round/scripts_laguna/run_laguna_s21_w4_experts_w8_rest.sh
```

The default output is
`work_dirs/Laguna-S-2.1-quarot-autoround-expert-w4-rest-w8-g64`. Rotation uses
logical `cuda:0`; AutoRound uses logical devices `0,1,2,3` after the launcher
sets `CUDA_VISIBLE_DEVICES=4,5,6,7`.

Common overrides:

```bash
AR_ITERS=50 AR_NSAMPLES=32 AR_BATCH_SIZE=1 \
  bash third_party/auto-round/scripts_laguna/run_laguna_s21_w4_experts_w8_rest.sh \
  /path/to/Laguna-S-2.1 /path/to/output
```

Set `AR_VERIFY_CHAT=0` to skip the post-export generation check. The verifier
defaults to `auto_round:tritonv2_zp`, which is compatible with the exported
`auto_round:auto_gptq` packing and does not require the optional Marlin CUDA
extension. Override it with `AR_INFERENCE_BACKEND` when another compatible
backend is installed. Verification uses the checkpoint's own sampling and stop
configuration; `AR_VERIFY_PROMPT`, `AR_VERIFY_MAX_NEW_TOKENS`, and
`AR_VERIFY_SEED` control only the request.

Rotation controls can be supplied through environment variables:

```bash
AR_ROTATION=hadamard \
AR_ROTATION_DEVICE=cuda:0 \
AR_ROTATION_COMPUTE_DTYPE=float64 \
AR_ROTATION_CHUNK_ROWS=1024 \
AR_ROTATION_CHUNK_COLUMNS=1024 \
AR_ROTATION_MATRIX_BATCH=4 \
  bash third_party/auto-round/scripts_laguna/run_laguna_s21_w4_experts_w8_rest.sh
```

Use `AR_ROTATION=none` only to reproduce the old AutoRound-only pipeline. To
save and inspect a rotated BF16 checkpoint without quantizing it, invoke
`quantize_laguna.py` directly with `--save-rotated PATH --rotation-only`.

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

After quantization, run a real generation check with:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH="$PWD/third_party/auto-round:$PWD" \
/data01/home/xuzk/anaconda3/envs/gptqmodel/bin/python \
  third_party/auto-round/scripts_laguna/chat_quantized.py \
  --model work_dirs/Laguna-S-2.1-quarot-autoround-expert-w4-rest-w8-g64 \
  --prompt "请用一句话介绍你自己。"
```

The verifier rejects unquantized checkpoints, checkpoints without Laguna's
QuaRot metadata, loads the quantized modules through the local AutoRound
Transformers backend, and fails if generation is empty or contains invalid
Unicode.
