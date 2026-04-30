# Qwen3.5 / Qwen3.5-MoE 旋转量化流程

本目录的 Qwen3.5 脚本用于完整 VLM checkpoint 的 GPTQ 量化：

- `example_qwen35dense.py`：Qwen3.5 Dense / VL，例如 `Qwen3.5-9B`
- `example_qwen35moe.py`：Qwen3.5 MoE / VL，例如 `Qwen3.5-35B-A3B`

当前流程默认同时处理 LLM、vision passthrough、MTP sidecar。打开 `--rotation` 后，LLM、vision merger、MTP 使用同一套旋转约定；保存后会写入 `mtp_rotated.safetensors`，并保留 vision tower 权重。

## 关键注意事项

1. **旋转量化必须关闭 disk offload。** `offload_to_disk=True` 已确认会导致 Qwen3.5 rotated quant checkpoint 推理乱码，因此两个脚本默认 `--no-offload-to-disk`，并且在 `--rotation` 下显式传 `--offload-to-disk` 会直接报错。
2. **建议 always 带 `--check-quant-vision-demo`。** 该检查会加载量化后的完整 VLM 模型，做图文 demo 对比；失败时会打印 max/mean logits diff、首个不同 generated token，以及 ref/test decoded 文本。
3. **已有量化输出会自动复用。** 如果 `--out` 已经是 GPTQ 目录，脚本默认跳过量化，直接执行后续推理/PPL/vision demo；但如果旧产物记录为 `offload_to_disk=true` 且当前带 `--rotation`，会强制重新量化，避免复用坏 checkpoint。
4. **校准 JSONL 每行一个 JSON 对象。** 默认读取 `text` 字段，可用 `--calibration-text-key` 修改字段名。
5. **多卡使用 `--device-map auto`。** Dense 9B 单卡通常可跑；MoE 35B-A3B 推荐多卡 auto。

## 输出命名规范

推荐统一放在 `work_dirs/` 下：

| 模型 | 正式输出 | Smoke/debug 输出 |
|---|---|---|
| Qwen3.5 Dense 9B W4G64 rotate | `work_dirs/q9b-W4G64-rot` | `work_dirs/q9b-W4G64-rot-smoke-debug` |
| Qwen3.5 MoE 35B-A3B W4G64 rotate | `work_dirs/q35b-W4G64-rot-v2` | `work_dirs/q35b-W4G64-rot-smoke-debug` |

如需迭代同一配置，用 `-v2`、`-v3` 后缀，不要覆盖已确认可用的目录。

## Dense 9B 推荐命令

正式量化命令：

```bash
python examples/quantization/examples/example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-9B \
  --rotation hadamard \
  --bits 4 \
  --group-size 64 \
  --nsamples 256 \
  --out work_dirs/q9b-W4G64-rot \
  --device-map auto \
  --calibration-jsonl gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl \
  --check-quant-vision-demo
```

Smoke/debug 命令：

```bash
python examples/quantization/examples/example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-9B \
  --rotation hadamard \
  --bits 4 \
  --group-size 64 \
  --nsamples 1 \
  --seqlen 64 \
  --batch-size 1 \
  --out work_dirs/q9b-W4G64-rot-smoke-debug \
  --no-offload-to-disk \
  --device cuda:0 \
  --check-quant-vision-demo
```

## MoE 35B-A3B 推荐命令

正式量化命令：

```bash
python examples/quantization/examples/example_qwen35moe.py \
  --model /data01/datasets/Qwen3.5-35B-A3B \
  --rotation hadamard \
  --bits 4 \
  --group-size 64 \
  --self-attn-bits 8 \
  --shared-expert-bits 8 \
  --expert-down-bits 5 \
  --nsamples 512 \
  --device-map auto \
  --out work_dirs/q35b-W4G64-rot-v2 \
  --check-quant-vision-demo \
  --calibration-jsonl gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl
```

MoE bit 配置约定：

| 模块 | 推荐 bits |
|---|---:|
| self-attn / linear-attn | 8 |
| shared expert | 8 |
| routed expert gate/up | 4 |
| routed expert down_proj | 5 |
| 其他 base linear | 4 |

## 校准数据

Dense 推荐使用：

```text
gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl
```

MoE 推荐使用：

```text
gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl
```

JSONL 示例：

```jsonl
{"text": "这是第一条校准文本。"}
{"text": "GPTQ quantization approximates weights with low-bit groups."}
```

## 验证项

脚本默认会在量化完成后执行：

1. 加载量化模型并做一次文本生成。
2. 计算 wikitext2 PPL。
3. 如果传入 `--check-quant-vision-demo`，执行 VLM 图文 demo 对比。

vision demo 失败时会打印类似：

```text
[vision] max_abs_diff=..., mean_abs_diff=..., last_token_equal=False, generation_equal=False
[vision] first_diff_generated_token=...
[vision] ref_decoded ='...'
[vision] test_decoded='...'
```

## 只复用已有量化模型

如果 `--out` 已存在且是 GPTQ 目录，脚本会自动跳过量化。也可以显式指定：

```bash
python examples/quantization/examples/example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-9B \
  --out work_dirs/q9b-W4G64-rot \
  --skip-quantize \
  --device-map auto \
  --check-quant-vision-demo
```

## 仅保存 rotated FP checkpoint

如果只需要验证/保存旋转后的 FP 模型：

```bash
python examples/quantization/examples/example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-9B \
  --rotation hadamard \
  --rotated-out work_dirs/q9b-rot-fp-full-verify \
  --check-rotation-ppl \
  --check-vision-demo \
  --stop-after-rotate \
  --no-offload-to-disk \
  --device cuda:0
```

MoE 使用 `example_qwen35moe.py`，并把 `--model` / `--rotated-out` 换成对应目录。
