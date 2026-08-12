# Qwen3.5 / Qwen3.5-MoE 旋转量化流程

本目录的 Qwen3.5 脚本用于完整 VLM checkpoint 的 GPTQ 量化：

- `example_qwen35dense.py`：Qwen3.5 Dense / VL，例如 `Qwen3.5-9B`
- `example_qwen35moe.py`：Qwen3.5 MoE / VL，例如 `Qwen3.5-35B-A3B`

当前流程默认同时处理 LLM、vision passthrough、MTP sidecar。打开 `--rotation` 后，LLM、vision merger、MTP 使用同一套旋转约定；保存后会写入 `mtp_rotated.safetensors`，并保留 vision tower 权重。

## 关键注意事项

1. **旋转量化必须关闭 disk offload。** `offload_to_disk=True` 已确认会导致 Qwen3.5 rotated quant checkpoint 推理乱码，因此两个脚本默认 `--no-offload-to-disk`，并且在 `--rotation` 下显式传 `--offload-to-disk` 会直接报错。
2. **Qwen3.5 rotation 与 MTP/spec-decode 不兼容。** GPTQModel 示例脚本在开启 `--rotation` 时会按同一旋转约定处理 LLM、vision merger 和 MTP sidecar；但 xh2modelzoo 的 MTP/spec-decode 导出链路不能消费 rotated MTP。因此面向 xh2modelzoo 的 workflow YAML 默认必须设置 `quant.rotation: false`。如需 GPTQModel rotated checkpoint，请显式关闭 xh2modelzoo 的 MTP/spec-decode 配置后再开启 rotation。
3. **建议 always 带 `--check-quant-vision-demo`。** 该检查会加载量化后的完整 VLM 模型，做图文 demo 对比；失败时会打印 max/mean logits diff、首个不同 generated token，以及 ref/test decoded 文本。
4. **保持 `--wait-for-submodule-finalizers` 默认开启。** GPTQ packing/finalize 是异步任务；Qwen3.5/Qwen3.6 在未等待 finalizer 时可能在 save 阶段写出全 0 的 `qweight/qzeros/scales`，表现为 demo 全 `!`、PPL 为 `nan`。只有排查性能问题时才显式使用 `--no-wait-for-submodule-finalizers`。
5. **已有量化输出会自动复用。** 如果 `--out` 已经是 GPTQ 目录，脚本默认跳过量化，直接执行后续推理/PPL/vision demo；但如果旧产物记录为 `offload_to_disk=true` 且当前带 `--rotation`，会强制重新量化，避免复用坏 checkpoint。
6. **校准 JSONL 每行一个 JSON 对象。** 默认读取 `text` 字段，可用 `--calibration-text-key` 修改字段名。
7. **设备映射使用 `--device-map auto`。** Dense 9B 和 MoE 35B-A3B 均可单卡运行；如需多卡再显式设置 `CUDA_VISIBLE_DEVICES` 暴露多张卡。

## 输出命名规范

推荐统一放在 `work_dirs/` 下：

| 模型                             | 正式输出                      | Smoke/debug 输出                       |
| -------------------------------- | ----------------------------- | -------------------------------------- |
| Qwen3.5 Dense 9B W4G64 rotate    | `work_dirs/q9b-W4G64-rot`     | `work_dirs/q9b-W4G64-rot-smoke-debug`  |
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

python examples/quantization/examples/example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-0.8B \
  --bits 4 \
  --group-size 64 \
  --nsamples 256 \
  --out work_dirs/q0.8b-W4G64 \
  --device-map auto \
  --calibration-jsonl gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl \
  --check-quant-vision-demo
```

非旋转量化不要传 `--rotation`；脚本的 argparse choices 只接受
`hadamard`、`block_hadamard`、`random` 三种真实旋转模式。

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

## MoE 122B-A10B 推荐命令

CUDA_VISIBLE_DEVICES=2 python examples/quantization/examples/example_qwen35moe.py \
 --model /data01/nfs_shared/llm_models/Qwen/Qwen3.5-122B-A10B \
 --bits 4 \
 --rotation hadamard \
 --group-size 64 \
 --self-attn-bits 8 \
 --shared-expert-bits 8 \
 --expert-down-bits 5 \
 --nsamples 512 \
 --no-offload-to-disk \
 --check-quant-vision-demo \
 --calibration-jsonl gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl

注意必须指定 --no-offload-to-disk，否则量化后的模型会出现乱码

MoE bit 配置约定：

| 模块                    | 推荐 bits |
| ----------------------- | --------: |
| self-attn / linear-attn |         8 |
| shared expert           |         8 |
| routed expert gate/up   |         4 |
| routed expert down_proj |         5 |
| 其他 base linear        |         4 |

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

# Ling-3-Flash GPTQ 量化

`example_ling3_flash.py` 通过
`gptqmodel.recipes.ling3_flash.quantize_ling3_flash` 执行 Ling-3-Flash
文本模型量化，固定部署精度为：普通 Linear/attention/shared expert W8、
routed experts W4、G64、对称量化；KDA decay/beta 投影保持 BF16。

当前已验证的 Ling recipe 使用 `offload_to_disk=False`，因此默认并推荐显式
关闭 disk offload。`--offload-to-disk` 仅保留给专项验证；启用后必须重新验证
生成 checkpoint 的完整性与推理结果。

```bash
CUDA_VISIBLE_DEVICES=0 python examples/quantization/examples/example_ling3_flash.py \
  --model /path/to/Ling-3.0-flash \
  --out work_dirs/ling-3-flash-gptq-w8w4-g64 \
  --calibration-jsonl /path/to/calibration.jsonl \
  --nsamples 128 \
  --seqlen 2048 \
  --batch-size 1 \
  --no-offload-to-disk \
  --wait-for-submodule-finalizers \
  --device cuda:0
```

xh2modelzoo 等下游调用方应直接使用 recipe API，不应自行构造 Ling 的
`QuantizeConfig`：

```python
from gptqmodel.recipes.ling3_flash import quantize_ling3_flash

quantize_ling3_flash(
    model_dir="/path/to/Ling-3.0-flash",
    output_dir="work_dirs/ling-3-flash-gptq-w8w4-g64",
    calibration_jsonl="/path/to/calibration.jsonl",
    offload_to_disk=False,
)
```

# Gemma4-26B-A4B量化使用说明

由于gemma4-moe结构的特殊性，在gptqmodel中无需旋转，因此，量化指令如下：

```
CUDA_VISIBLE_DEVICES=7 python examples/quantization/examples/example_gemma4.py --model /data01/home/chenzx/model/gemma-4-26B-A4B-it --out ~/model/gemma-4-26B-A4B-it-gptq-20260508 --calibration-jsonl ~/project/gptqmodel/gen_qwen3_30b_EBSS.jsonl --nsamples 512 --bits 4 --group-size 64 --no-offload-to-disk --skip-eval --batch-size 1
```

该例子中使用qwen3-30b的校准集，用户亦可使用gemma4的自生成，可参考`gptqmodel/quantization/calibration/README.md`中校准集生成方法
