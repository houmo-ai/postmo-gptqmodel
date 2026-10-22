# DeepSeek-V4-Flash-0731 W8/W4-G64

DeepSeek-V4 Flash 的原始 checkpoint 混合了 FP8 E4M3-B128×128 和 routed-expert E2M1-FP4-B32，不能直接作为普通 BF16 GPTQ/AutoRound 输入。示例会按 shard 流式反量化，并支持物理裁剪前 N 层。部署 profile 固定为：非共享 routed experts W4，其余已登记的普通量化 Linear（包括 shared expert 和主 attention 投影）W8，group size 64，对称量化；Grouped `o_a`、compressor/indexer、router、norm、mHC 保持浮点。

前 6 层（覆盖 SWA、CSA、HCA）的 GPTQ 推荐命令：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n xhquant_55 \
python examples/quantization/examples/example_deepseek_v4_flash.py \
  --model /data01/datasets/DeepSeek-V4-Flash-0731 \
  --prepared-model work_dirs/deepseek-v4-flash-bf16-first6 \
  --out work_dirs/deepseek-v4-flash-first6-GPTQ-W8E4G64 \
  --num-layers 6 \
  --method gptq \
  --bits 8 \
  --expert-bits 4 \
  --group-size 64 \
  --nsamples 128 \
  --seqlen 2048
```

AutoRound 使用相同的 mixed-precision 规则：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n xhquant_55 \
python examples/quantization/examples/example_deepseek_v4_flash.py \
  --model /data01/datasets/DeepSeek-V4-Flash-0731 \
  --prepared-model work_dirs/deepseek-v4-flash-bf16-first6 \
  --out work_dirs/deepseek-v4-flash-first6-AutoRoundV2-W8E4G64 \
  --num-layers 6 \
  --method autoround \
  --auto-round-version v2 \
  --auto-round-iters 200
```

两条路径均默认 `offload_to_disk=False`；只有显式添加 `--offload-to-disk` 才会创建 offload 目录。GPTQ 和 AutoRound 按 decoder layer、subset、MoE module chunk 串行推进，实测单层峰值约 24 GiB，因此默认只暴露一张 80-GiB GPU。`auto_forward_data_parallel` 也默认关闭，避免把当前层复制到每张可见 GPU；暴露多卡只会提供可选 expert worker pool，并不是容量要求。完整 43 层去掉 `--num-layers 6`，同样使用单卡，并为 BF16 staging 和最终量化产物预留足够空间。前三层虽然使用 `tid2eid` Hash 路由，但 256 个真实 expert 的 gate/up/down 仍统一走 W4。该流程保留 latent/shared-KV 与 grouped `o_a/o_b` 分解，不会展开传统 K/V 投影，也不会合并 `o_a/o_b`。

## FP4 → INT4 权重误差审计

`audit_deepseek_v4_fp4_to_int4.py` 不生成新 checkpoint，而是直接按 shard
读取原生 routed-expert 权重并流式统计。0731 checkpoint 的真实布局是：每个
signed byte 按 low-nibble-first 存两个 E2M1 数值，`Float8_e8m0fnu` scale 是
直接乘数，且每个 scale 精确覆盖 32 个逻辑输入列。原始 G32 block 可以直接
沿用，G64 则必须合并相邻两个原生 block。

单卡小样：

```bash
python examples/quantization/examples/audit_deepseek_v4_fp4_to_int4.py \
  --model /data01/datasets/DeepSeek-V4-Flash-0731 \
  --output work_dirs/deepseek-v4-fp4-int4-smoke.json \
  --tensor-records work_dirs/deepseek-v4-fp4-int4-smoke.jsonl \
  --rows-per-tensor 64 \
  --devices cuda:0 \
  --max-tensors 3 \
  --lloyd-iterations 3
```

四卡覆盖全部 33,024 个 routed-expert tensor、每个 tensor 均匀抽 64 行：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/quantization/examples/audit_deepseek_v4_fp4_to_int4.py \
  --model /data01/datasets/DeepSeek-V4-Flash-0731 \
  --output work_dirs/deepseek-v4-fp4-int4-all-r64.json \
  --tensor-records work_dirs/deepseek-v4-fp4-int4-all-r64.jsonl \
  --rows-per-tensor 64 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --lloyd-iterations 3
```

脚本同时报告普通 RTN、baicai1145 公布的六候选 scale 网格以及多起点
Lloyd/least-squares scale refinement。Lloyd 的每轮先固定 scale 计算
`q=clamp(round(w/s),-8,7)`，再固定 `q` 更新
`s=sum(w*q)/sum(q*q)`；最后以实际存储为 FP16 后的重构误差选择最优起点。
`--rows-per-tensor 0` 会对命中的权重逐元素全量计算，适合复核少数离群 expert。

若只下载了 baicai1145 W4A16 的单个 safetensor shard，可逐元素验证成品
packing 和量化算法，无需下载完整模型：

```bash
python examples/quantization/examples/verify_deepseek_v4_w4a16_tensor.py \
  --native-model /data01/datasets/DeepSeek-V4-Flash-0731 \
  --converted-shard /path/to/model-00020-of-00034.safetensors \
  --tensor layers.29.ffn.experts.126.w1.weight \
  --device cuda:0 \
  --output work_dirs/baicai-layer29-expert126-w1-verification.json
```

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

## Qwen3-VL 特征编码器

GPTQ / AutoRound 的公开 recipe、原生多模态校准和 ModelZoo 消费方式见 [Qwen3-VL 特征量化](qwen3_vl_features.md)。
