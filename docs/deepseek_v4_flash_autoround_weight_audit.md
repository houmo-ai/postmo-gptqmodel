# DeepSeek-V4-Flash AutoRound packed 权重审计

## 结论

当前 `repacked-clamped-v2` checkpoint 的量化辅助数据和模块结构是完整、合法的；
layer 29 的权重误差也没有表现出层特异性异常，尤其是运行时诊断中命中的
expert 126 并不是权重离群点。因此，layer 29 激活突增不能归因于
`repacked-clamped-v2` 内某个坏掉的 qweight、scale、zero-point 或 `g_idx`，后续应
重点检查 packed runtime 的激活 dtype、mHC/FFN residual、MoE 路由与量化 Linear
执行路径。

同时，修复前的 original artifact 确实存在严重 bit-pack 损坏。以 layer 29
expert 126 为例，修复使三张 W4 权重的相对 RMSE 从 59%～62% 降到 9.1%～9.5%，
说明 clamp + repack 修复是必要且有效的，不能回退到 original artifact。

## 审计对象和映射

- packed：
  `work_dirs/deepseek-v4-flash-full43-AutoRoundV2-W8E4G64-smoke-n1-s16-i1-nooffload-repacked-clamped-v2`
- 修复前 packed：
  `work_dirs/deepseek-v4-flash-full43-AutoRoundV2-W8E4G64-smoke-n1-s16-i1-nooffload`
- 官方 native 权重展开得到的 BF16 staging：
  `work_dirs/deepseek-v4-flash-bf16-full`

packed index 中共有 33,325 个量化 Linear。审计脚本按 `quantization_config.dynamic`
使用和 GPTQModel 一致的 `re.match` 规则解析精度：

| 类型 | tensor 数 | 配置 |
|---|---:|---|
| routed expert | 33,024 | W4、group size 64、symmetric |
| attention + shared expert | 301 | W8、group size 64、symmetric |

BF16 staging 使用官方 native 命名，packed 使用 Transformers 命名。审计没有按
同名猜测，而是复用 checkpoint repair 的映射：

| packed projection | native source |
|---|---|
| `gate_proj` | `w1.weight` |
| `down_proj` | `w2.weight` |
| `up_proj` | `w3.weight` |
| `q_a_proj` | `wq_a.weight` |
| `q_b_proj` | `wq_b.weight` |
| `kv_proj` | `wkv.weight` |
| `o_b_proj` | `wo_b.weight` |

映射命中 33,325/33,325，未映射 0，BF16 source 缺失 0。量化辅助 tensor 可能
跨 safetensor shard，脚本严格按 index 分别读取 `qweight/qzeros/scales/g_idx`。

GPTQ-v1 qzero 的解码按完整 uint32 word 加 packed-one offset，再按 bit slot
展开，和运行时 v1→v2 转换一致。正常 repaired artifact 使用每模块 bit 宽对应的
offset；已知 original mixed W8/W4 artifact 使用错误的 global W8 offset，原始对照
扫描显式使用 `--qzero-source-offset global`，避免把已知 offset bug 和 qweight
pack bug 混在一起。

## 全模型辅助数据审计

这一项覆盖全部 33,325 个量化 Linear，不是抽样：

| 检查项 | 非法元素数 |
|---|---:|
| module/source 映射失败 | 0 |
| companion tensor 缺失或 shape 错误 | 0 |
| scale 非有限值 | 0 |
| scale 非正值 | 0 |
| zero-point 超出 bit 范围 | 0 |
| symmetric zero-point 不等于中点 | 0 |
| `g_idx` 越界 | 0 |
| `g_idx` 与 group-size 64 布局不一致 | 0 |

结果文件：

- `full_aux_structure_audit.csv`：33,325 行逐 tensor 明细；
- `full_aux_structure_audit.json`：汇总和异常计数。

两者位于 repaired checkpoint 目录。

## layer 28 与 layer 29 全元素反量化误差

每层均完整扫描 775 个 Linear；每张权重的所有元素都参与 RMSE、SNR、cosine、
mean bias 和 max error 计算。

### Routed expert W4

| 层 | tensor 数 | rel-RMSE mean | p95 | p99 | max | cosine mean |
|---:|---:|---:|---:|---:|---:|---:|
| 28 | 768 | 0.09563 | 0.10711 | 0.11330 | 0.12046 | 0.99541 |
| 29 | 768 | 0.09629 | 0.10936 | 0.11375 | 0.11636 | 0.99535 |

两层分布几乎重合。layer 29 最大 rel-RMSE 是 expert 72 `gate_proj` 的 0.11636，
属于常规 W4 误差量级，并不存在单张权重突然达到几十倍误差的现象。

### layer 29 W8

| projection | rel-RMSE | cosine |
|---|---:|---:|
| attention `q_a_proj` | 0.00597 | 0.999982 |
| attention `q_b_proj` | 0.00585 | 0.999983 |
| attention `kv_proj` | 0.00651 | 0.999979 |
| attention `o_b_proj` | 0.02088 | 0.999782 |
| shared `down_proj` | 0.02260 | 0.999745 |
| shared `up_proj` | 0.04365 | 0.999048 |
| shared `gate_proj` | 0.06246 | 0.998048 |

shared expert 的误差比 attention W8 大，主要由权重动态范围和少量大值驱动；但
layer 29 并不异常。全 43 层相同 projection 的对照见下一节。

## 全模型 301 张 W8 权重

全部 301 张 W8 权重均完成全元素扫描：

| family/projection | tensor 数 | rel-RMSE mean | p95 | max |
|---|---:|---:|---:|---:|
| attention `q_a_proj` | 43 | 0.00746 | 0.01003 | 0.01724 |
| attention `q_b_proj` | 43 | 0.00589 | 0.00594 | 0.00646 |
| attention `kv_proj` | 43 | 0.01423 | 0.02891 | 0.04476 |
| attention `o_b_proj` | 43 | 0.03144 | 0.04448 | 0.07428 |
| shared `gate_proj` | 43 | 0.05382 | 0.07529 | 0.08735 |
| shared `up_proj` | 43 | 0.04670 | 0.09520 | 0.09980 |
| shared `down_proj` | 43 | 0.03052 | 0.06190 | 0.13036 |

全 W8 最大 rel-RMSE 是 layer 42 shared `down_proj` 的 0.13036，而不是 layer 29。
layer 29 的七张 W8 权重均处于各自 projection 的正常分布范围。

结果文件位于 repaired checkpoint 目录：

- `all_w8_full_exact_weight_audit.csv/json`；
- `layer28_full_exact_weight_audit.csv/json`；
- `layer29_full_exact_weight_audit.csv/json`。

## expert 126：修复前与修复后

以下每一行都使用完整的 8,388,608 个权重元素：

| projection | artifact | rel-RMSE | SNR/dB | cosine | max abs error |
|---|---|---:|---:|---:|---:|
| down | original | 0.59374 | 4.53 | 0.82438 | 0.36523 |
| down | repaired | 0.09065 | 20.85 | 0.99590 | 0.12493 |
| gate | original | 0.61610 | 4.21 | 0.81044 | 0.25391 |
| gate | repaired | 0.09325 | 20.61 | 0.99566 | 0.12447 |
| up | original | 0.61882 | 4.17 | 0.80878 | 0.25000 |
| up | repaired | 0.09537 | 20.41 | 0.99546 | 0.12402 |

在 repaired layer 29 内，expert 126 `gate_proj` 的误差是 256 个专家里第 5 低，
`up_proj` 第 15 低；`down_proj` 排第 202/256，但仍在连续正常分布内。因此运行时
命中 expert 126 不能解释为 expert 126 权重文件损坏。

## GPTQModel runtime 审计（2026-08-13）

### 硬结论

`repacked-clamped-v2` 已在 8 张 A100 上用 GPTQModel Triton runtime 完整
load，没有 CPU/disk offload，并成功完成精确 8,192 token prefill 和后续
decode。因此，checkpoint 不是“只能解析、不能运行”的结构产物。

512 token one-shot 与 `2 x 256` chunked prefill **不是严格数值等价**：
即使在整段运行中显式设置
`allow_fp16_reduced_precision_reduction=False`，最后一个 token 的 logits 仍是
cosine 0.999494、rel-RMSE 3.1709%，未通过现有 1% rel-RMSE 门限。
但分布层面的差异明显小于 raw-logit 相对误差：全序列 512 个位置中
509 个 top-1 相同（99.4141%），平均 JS divergence 为 0.0001596，同一
prompt 的 next-token PPL 为 1.86797 对 1.87674，相差约 0.47%。所以硬结论
是“不能声称数值等价，但没有证据表明 chunk cache/位置逻辑失效”。

### 512 token 全序列对照

对照使用同一 checkpoint、同一 FP16 autocast 和两个全新
`DynamicCache`：一路单次 forward 512 token，另一路 forward `256 + 256`。
两路都返回完整 `[1, 512, 129280]` logits，然后按 token 在 CPU 上用
FP32 计算统计量，不只比较最后一行。

| 指标 | 结果 |
|---|---:|
| 两路 finite logits | 100% |
| cache 最终长度 | 512 / 512 |
| 全序列 top-1 相同 | 509/512 = 99.4141% |
| token-wise cosine mean / p50 / p95 / min | 0.996131 / 0.998161 / 0.999665 / 0.933409 |
| token-wise rel-RMSE mean / p50 / p95 / max | 7.4659% / 6.1296% / 16.6183% / 40.1811% |
| KL(one-shot || chunked) mean / p95 / max | 0.0006362 / 0.0044255 / 0.0261869 |
| KL(chunked || one-shot) mean / p95 / max | 0.0006468 / 0.0044190 / 0.0282014 |
| JS divergence mean / p95 / max | 0.0001596 / 0.0011162 / 0.0067528 |
| next-token mean NLL | 0.624851 / 0.629538 |
| mean NLL abs / relative difference | 0.004687 / 0.7501% |
| next-token PPL | 1.867967 / 1.876743（chunked 高约 0.47%） |
| per-token NLL abs difference mean / p95 / max | 0.013396 / 0.053697 / 1.203125 |

NLL/PPL 覆盖 prompt 内部 511 个 next-token transition。这个 prompt 不是标准
perplexity benchmark：它是 demo 生成的确定性中性中文 filler，末尾再放入真实
问题和 assistant generation suffix；重复 filler 会让绝对 PPL 偏低。这些数值
只能用于同一 prompt/checkpoint 下比较 one-shot 与 chunked 执行调度，不能
解读为模型在标准语料上的 PPL，也不能用于对比 BF16 基线质量。

严格运行证据：

- `fp16_reduced_precision_reduction` 记录的 requested/actual 均为 `false`，
  退出审计上下文后成功恢复原值 `true`；
- 8 张 GPU 合计驻留 161,072,287,612 bytes 模型 tensor，
  `non_cuda_tensors=[]`；
- one-shot/chunked 的 chunk 列表分别是 `[512]` 和 `[256, 256]`，
  cache 长度都是 512；
- 逐层诊断中 463 项 cache 对照全部 shape 一致，148 项标量状态
  全部相等。张量值随上游浮点差异而不完全相等，但没有长度、
  entry count 或 buffer shape 错位。

结果文件：

- `gptqmodel-triton-fp16-strict-reduction-chunk256-512-full-sequence-equivalence.json`；
- `gptqmodel-triton-fp16-strict-reduction-chunk256-512-layer-cache-diagnostic.json`。

两者都在 repaired checkpoint 目录下。前者 SHA-256 为
`99dbb1dbe1bad0f510db9443b87deaa8d3eb9d42d7c06eb579d645606963d8e1`。

### 逐层与 layer 0 GEMM replay 定位

严格 reduction 运行的 layer 0 差异已在很小的浮点尺度出现：
`self_attn` rel-RMSE 0.0329%，layer 0 输出 rel-RMSE 0.0324%；经 43 层
递推后才放大成最后 logits 的 3.17%。例如 layer 10/20/29 的最大阶段
rel-RMSE 分别为 1.14%/5.73%/3.62%。这是逐层放大，不是突然从
layer 29 开始的非有限值或结构错误。

`layer0-kv-proj-m-shape-replay.json` 对同一组 layer 0 `kv_proj` 输入分别用
`M=512`、`M=256` 和 `M=1` replay：

- packed Triton 输出与固定反量化权重的 dense FP16 输出在 `M=512`、
  `M=256` 下均逐元素完全相等；
- 但无论 packed 还是 fixed-weight dense，`M=512` 的后 256 行与单独
  `M=256` 计算均只有 52.3193% 元素完全相等，rel-RMSE 0.04028%，
  max abs error 0.00390625；
- 最后一行 `M=512` 对 `M=256` 的 rel-RMSE 为 0.04287%，
  `M=512` 对 `M=1` 为 0.03483%。

这个 replay 排除了“不同 M 下 Triton 解 pack/反量化了不同权重”；差异
在固定 dense 权重下同样存在，属于 GPU FP16 GEMM 因 M shape/算法选择和
累加顺序不同而产生的舍入差异。关闭 reduced-precision reduction 能改变早期
差异量级，但不能保证不同 M 的 GEMM bitwise 相同。

这与 layer 29 expert 126 replay 的结论一致：捕获的 runtime 输出与
Triton current path 完全一致，packed dense 对 Triton actual down 的 rel-RMSE
只有 0.0191%；它不是 expert 126 的特殊 kernel 故障。

### 8K prefill + decode 实跑

`gptqmodel-triton-fp16-chunk256-8k-512.json` 的 prompt 精确为 8,192 token，
其中 8,172 token 为确定性 filler，末尾保留中文问题和 generation suffix。
prefill 实际为 32 次 `256` token forward，8 张 GPU 完全驻留，所有保留
score 均 100% finite。用 `temperature=1.0`、最多 512 个新 token 采样，模型
在第 160 个 token 生成 EOS；160 个 token 有 99 个 unique ID，最长连续
重复为 1，生成了语义正常的中文“瑞利散射解释蓝天”答案，不是
重复 token 退化。本次实测 prefill 为 59.13 token/s，decode 为 1.262
token/s；这是 GPTQModel/Transformers eager runtime 的单次观测，不是 vLLM 或
TRT-LLM 性能基准。

## 审计脚本

脚本：
`examples/quantization/examples/audit_deepseek_v4_packed_checkpoint.py`

它支持：

- dynamic W4/W8 解析和 native→packed 映射覆盖检查；
- GPTQ-v1 qzero 完整 word 解码；
- 流式、多 GPU、逐 tensor 全元素反量化；
- rel-RMSE、RMSE、SNR、cosine、mean bias、max error；
- aux-only 全模型快速检查；
- prefix sample 快速排查；
- CSV 逐 tensor 结果与 JSON 分布、Top-100 outlier；
- 每 worker 限制 CPU 线程，safetensor reader 使用最多 8 shard 的 LRU，避免长期
  扫描累计高 RSS。

全模型 exact 命令如下，整网 demo 空闲后可继续执行：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
DEEPSEEK_AUDIT_CPU_THREADS=4 \
/data01/home/yujy/miniconda3/envs/fp8_nlt/bin/python -u \
  examples/quantization/examples/audit_deepseek_v4_packed_checkpoint.py \
  --quant-model work_dirs/deepseek-v4-flash-full43-AutoRoundV2-W8E4G64-smoke-n1-s16-i1-nooffload-repacked-clamped-v2 \
  --bf16-model work_dirs/deepseek-v4-flash-bf16-full \
  --output work_dirs/deepseek-v4-flash-full43-AutoRoundV2-W8E4G64-smoke-n1-s16-i1-nooffload-repacked-clamped-v2/full_exact_weight_audit \
  --devices cuda:0,cuda:1
```

本轮曾启动全 33,325 tensor 的全元素扫描，但为整网 demo 释放 CPU/I/O/GPU 后已
中断；`full_exact_weight_audit.part-*.jsonl` 仅是 7,452 tensor 的中间文件，不是
完整报告，不能据此声称全模型 W4 已完成全元素扫描。当前已完成且可用于结论的
范围是：全 33,325 tensor aux/结构、全 301 tensor W8、layer 28/29 全部权重，以及
expert 126 修复前后对照。

## vLLM 可加载性边界与 CPU-only dry-run

### 结论

当前 `repacked-clamped-v2` 不能被 vLLM 原样加载。问题不在 Marlin 是否支持
W4/W8 或 group size 64，而在 DeepSeek-V4 NVIDIA loader、checkpoint schema 和
`wo_a` 执行路径尚未对齐：

- vLLM 的 AutoGPTQ/Marlin 本身支持 symmetric W4、symmetric W8、group size 64，
  也允许同一个 checkpoint 的普通 Linear 使用 W8、整个 routed-expert MoE 使用
  W4；
- 一个 `RoutedExperts` 内必须使用同一种量化方案。本模型全部 routed expert 都是
  W4，因而满足这个限制；它没有要求“一个 expert 一种 bit”；
- 当前 checkpoint 的 W4/W8 数值布局可以保留，不需要先反量化再重量化；
- 但仅改启动参数或仅重命名 checkpoint 都不够。至少需要 vLLM source patch，
  尤其是 NVIDIA DSV4 fused-module 映射和 BF16 `wo_a` 路径。

因此当前建议分成两条路线：

1. **路线 A：先用 GPTQModel/Transformers 完成 8K demo 和精度定位。** 这是当前
   最短路径，checkpoint 使用的正是 HF module 命名，而且 Transformers 已有 BF16
   grouped `o_a_proj` 计算；
2. **路线 B：把 vLLM 作为独立适配任务。** 推荐“少量 checkpoint JSON 修正 +
   vLLM source patch”，不要先重写约 160 GB safetensor。只有在不愿维护 HF→native
   mapper 时，才做流式 key 转换；即使做了转换，`packed_modules_mapping` 和 BF16
   `wo_a` 两个 source patch 仍然不可省。

### 调研版本与环境

| 对象 | 实际状态 |
|---|---|
| 本地 `/data01/home/yujy/work/vllm` | `069ba34ed2`，约为 `v0.23.1rc0-1391`，相对 `origin/main` behind 895 |
| 最新只读 worktree `/data01/home/yujy/work/vllm_dsv4_gptq` | `61826c1c6f`，与当时 `origin/main` 一致，约为 `v0.27.2rc0-10` |
| conda `vllm_dsv4_gptq` 的已安装 wheel | `vllm 0.24.0+cu129`、Torch `2.11.0+cu129`、Transformers `5.13.0`、compressed-tensors `0.17.0` |

dry-run 使用最新 worktree 的 Python 源码叠加已安装 wheel 的二进制扩展，只在
CPU/meta device 上做 config、量化布局和 loader 路由检查；没有占用 GPU，也没有
声称这等价于完整 CUDA 推理。正式运行必须重新构建/安装与最新 source commit
匹配的 vLLM，不能长期混用 0.24 wheel 和 0.27 源码。

### 原样 checkpoint 的确定失败顺序

#### 1. `ModelConfig` 首先被 `hash_moe` 拒绝

当前 config 的前三个 `mlp_layer_types` 是 `hash_moe`。Transformers 5.13 的通用
layer-type validator 不包含这个枚举，因此原样执行 `ModelConfig` 的第一个错误是：

```text
StrictDataclassClassValidationError:
The `mlp_layer_types` entries must be in (..., 'moe', ...)
but got ['hash_moe', 'hash_moe', 'hash_moe', 'moe', ...]
```

vLLM DSV4 实际通过 `num_hash_layers` 决定 hash MoE，而当前 config 又没有
`num_hash_layers`。建议在 vLLM 的 `DeepseekV4Config` 兼容层中由
`mlp_layer_types` 推导 `num_hash_layers=3`，并把内部用于通用校验的三个
`hash_moe` 规范成 `moe`；不要为了 vLLM 直接破坏 GPTQModel 共用的原始 config。

#### 2. 绕过首错后还有三处 config schema 不兼容

当前 checkpoint 使用新版 Transformers schema，最新 vLLM DSV4 实现仍主要读取
官方 native/baicai schema：

| checkpoint 字段 | vLLM 当前读取 | dry-run 结果 |
|---|---|---|
| `layer_types` + `compress_rates` | `compress_ratios[layer_id]` | `AttributeError: ... no attribute 'compress_ratios'` |
| 嵌套 `rope_parameters.main/compress` | 单个 flat `rope_parameters` | 传入 RoPE cache key 时出现 `TypeError: unhashable type: 'dict'` |
| 前三项 `mlp_layer_types=hash_moe` | `num_hash_layers` | `AttributeError: ... no attribute 'num_hash_layers'` |

正确兼容方式是：

- 从 `layer_types` 和 `compress_rates` 推导每层 `compress_ratios`；
- RoPE 构造时按层选择 `main` 或 `compress` 子配置，而不是粗暴地永久丢掉其中一份；
- 推导 `num_hash_layers=3`。

此外，DSV4 的 `fp8_ds_mla` cache 路径要求显式使用 FP8 KV cache。dry-run 在
`cache_dtype=auto` 下会报：

```text
DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto
```

这不是 checkpoint 损坏，正式启动需要 `--kv-cache-dtype fp8`。

#### 3. AutoGPTQ 能识别 config，但量化模块识别错位

当前 `quant_method` 已修成 `gptq`，因此最新 vLLM 能将它解析为
`AutoGPTQConfig`：base W8、group 64、symmetric，dynamic routed expert W4。
旧 artifact 曾使用 `auto_round`，那会在量化方法选择阶段失败；当前版本已经不再有
这个首错。

不过，AutoGPTQ 默认通过“所有非浮点 safetensor”推导量化 module。实际得到
33,328 个 module，而真正的 `qweight` 只有 33,325 个，多出的三个正是 layer 0～2
的整数 `mlp.gate.tid2eid`。HF→native 映射完成后，这三个假阳性会把 BF16 router
gate 错建成 GPTQ Linear。

修复可二选一：

- 在 AutoGPTQ metadata 扫描中只接受 `*.qweight` 对应的 module；或
- 在 `quantize_config.json` 显式写入量化 module 类别，避免从 dtype 猜测。

第一种更接近根因修复。它应得到恰好 33,325 个量化 module。

#### 4. NVIDIA DSV4 loader 仍只接受 native namespace

当前 checkpoint 是 HF namespace，例如：

```text
model.layers.0.self_attn.q_a_proj.qweight
model.layers.0.mlp.experts.0.gate_proj.qweight
```

最新 NVIDIA DSV4 loader 内部及其现有 mapper 仍按 native namespace 工作，例如：

```text
layers.0.attn.wq_a.qweight
layers.0.ffn.experts.0.w1.qweight
```

单 tensor loader dry-run 的确定错误为：

```text
HF key     -> KeyError: 'layers.0.self_attn.q_a_proj.qweight'
native key -> 继续映射到 layers.0.attn.fused_wqa_wkv.qweight
```

后一个 native key 仍失败，是因为 fused module 被错误建成了 unquantized，而不是
native key 本身不被 loader 识别。

#### 5. NVIDIA class 缺少 fused GPTQ mapping

同一份最新源码中，AMD `DeepseekV4ForCausalLM` 已定义：

```python
packed_modules_mapping = {
    "gate_up_proj": ["w1", "w3"],
    "fused_wqa_wkv": ["wq_a", "wkv"],
}
```

NVIDIA class 没有该字段。因此 AutoGPTQ 不知道 checkpoint 的 `wq_a+wkv` 对应
运行时 `fused_wqa_wkv`，也不知道 shared expert 的 `w1+w3` 对应
`gate_up_proj`。

未修复 meta 构造的实际模块为：

| 内部模块 | 错误构造结果 |
|---|---|
| `attn.fused_wqa_wkv` | `UnquantizedLinearMethod` |
| `attn.wq_b`、`attn.wo_b` | `UnquantizedLinearMethod`，因为 HF/native 名称不匹配 |
| `ffn.shared_experts.gate_up_proj/down_proj` | `UnquantizedLinearMethod` |
| `ffn.experts` | `AutoGPTQMoEMethod`，但错误使用 base W8 |

必须把上述 mapping 加到 NVIDIA class，并保证 HF→native mapper 同时作用于
checkpoint weight key 和 `modules_in_block_to_quantize`。

#### 6. routed expert 的 dynamic rule 匹配不到 fused MoE

checkpoint dynamic rule 是：

```text
.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$ -> W4/G64
```

vLLM 创建 fused MoE 时只用整体 prefix：

```text
model.layers.L.ffn.experts
```

因此原规则永远不匹配，MoE 按 base W8 分配。未修复 layer 0 的实际布局是：

```text
w13_qweight [256, 1024, 4096]  # W8
w2_qweight  [256,  512, 4096]  # W8
```

而 checkpoint 单个 W4 `w1.qweight` 是 `[512, 2048]`。meta loader 甚至可能接受
这个较小 tensor，但它只覆盖 W8 目标切片的一半，不能把“loader 没立即抛错”当作
成功。

应保留原 HF dynamic rule供 GPTQModel 使用，并额外加入 vLLM fused rule：

```json
".*\\.ffn\\.experts$": {
  "bits": 4,
  "group_size": 64,
  "sym": true
}
```

模拟该规则和 packed mapping 后，meta 构造恢复成：

```text
w13_qweight [256, 512, 4096]   # W4
w2_qweight  [256, 256, 4096]   # W4
w13_qzeros  [256,  64,  512]
w2_qzeros   [256,  32,  512]
```

这与每 expert 的 W4 qweight/qzero/scale/g_idx 形状一致。

#### 7. BF16 `o_a_proj` 是无法靠 key 转换解决的硬 blocker

当前 43 个 `o_a_proj` 都是 BF16，设计上没有 GPTQ scale。vLLM 能把它构造成
unquantized `wo_a`，但 NVIDIA `_o_proj` 当前无条件执行：

```text
inverse RoPE + FP8 quant(o)
-> 读取 wo_a.weight_scale 或 wo_a.weight_scale_inv
-> fp8_einsum
```

BF16 `wo_a` 两种 scale 属性都不存在，因此即使权重已经成功加载，第一次 forward
仍会失败。此前 compressed-tensors 的修复只是在 `weight_scale` 与
`weight_scale_inv` 之间 fallback，不是 BF16 fallback。

最小修复可以复用/下沉 ROCm 已有的 BF16 reference 路径：

```text
inverse RoPE in BF16
-> o.view(T, local_groups, group_input_dim)
-> einsum/bmm("tgd,grd->tgr", o, wo_a_bf16)
-> flatten groups
-> wo_b W8
```

这个分支应由“`wo_a` 没有 FP8 scale 且 weight 是 BF16”静态选择，并与
Transformers `DeepseekV4GroupedLinear` 做逐层输出对比。不要给 BF16 weight 人工补
一个 scale=1 后仍喂给 `fp8_einsum`，那不是合法等价实现。

### 模拟最小修复后的构造与单 tensor load

CPU/meta dry-run 临时模拟了以下四项，但没有改共享 vLLM 源码：

1. 只从 33,325 个 `qweight` 生成 quantized-module 列表；
2. HF module 名转成 native module 名；
3. 加入 NVIDIA `packed_modules_mapping`；
4. 加入 fused expert W4 dynamic rule。

结果如下：

| 模块 | 修复后量化方法 |
|---|---|
| `fused_wqa_wkv` | AutoGPTQ W8/G64 |
| `wq_b`、`wo_b` | AutoGPTQ W8/G64 |
| shared `gate_up_proj/down_proj` | AutoGPTQ W8/G64 |
| routed `RoutedExperts` | AutoGPTQ W4/G64 |
| `wo_a` | unquantized BF16，符合 checkpoint |

使用 native key 的四个代表 tensor 均命中 loader：`wq_a.qweight`、routed
`w1.qweight`、shared `w1.qweight`、`wo_b.qweight`。这说明 GPTQ payload 无需离线
融合或 Marlin repack；vLLM loader 会把逐 expert `w1/w3` 写入 fused `w13_*`，把
`w2` 写入 `w2_*`，加载后再执行自身的 Marlin 转换。离线 converter 只应重命名
key，不能提前把 expert 拼接成另一种 qweight 布局。

### HF→native key 的最小转换规则

如果选择 checkpoint converter，而不是在 vLLM 增加 HF mapper，至少需要以下
规则；`qweight/qzeros/scales/g_idx` 后缀原样保留：

| HF | native |
|---|---|
| `model.layers.L.input_layernorm` | `layers.L.attn_norm` |
| `model.layers.L.post_attention_layernorm` | `layers.L.ffn_norm` |
| `model.layers.L.attn_hc.{fn,base,scale}` | `layers.L.hc_attn_{fn,base,scale}` |
| `model.layers.L.ffn_hc.{fn,base,scale}` | `layers.L.hc_ffn_{fn,base,scale}` |
| `model.layers.L.self_attn.{q_a_proj,q_b_proj,kv_proj,o_a_proj,o_b_proj}` | `layers.L.attn.{wq_a,wq_b,wkv,wo_a,wo_b}` |
| `model.layers.L.self_attn.q_a_norm` | `layers.L.attn.q_norm` |
| `model.layers.L.self_attn.sinks` | `layers.L.attn.attn_sink` |
| `model.layers.L.mlp.gate.e_score_correction_bias` | `layers.L.ffn.gate.bias` |
| `model.layers.L.mlp.experts.E.{gate,down,up}_proj` | `layers.L.ffn.experts.E.{w1,w2,w3}` |
| `model.layers.L.mlp.shared_experts.{gate,down,up}_proj` | `layers.L.ffn.shared_experts.{w1,w2,w3}` |
| `model.hc_head.{hc_fn,hc_base,hc_scale}` | `{hc_head_fn,hc_head_base,hc_head_scale}` |
| `model.embed_tokens.weight` | `embed.weight` |
| `model.norm.weight` | `norm.weight` |
| `lm_head.weight` | `head.weight` |

Compressor/Indexer 不能只做全局字符串替换，需按结构映射：

| HF | native |
|---|---|
| `self_attn.compressor.{kv_proj,gate_proj,kv_norm,position_bias}` | `attn.compressor.{wkv,wgate,norm,ape}` |
| `self_attn.compressor.indexer.{q_b_proj,scorer.weights_proj}` | `attn.indexer.{wq_b,weights_proj}` |
| `self_attn.compressor.indexer.{kv_proj,gate_proj,kv_norm,position_bias}` | `attn.indexer.compressor.{wkv,wgate,norm,ape}` |

converter 必须重写每个 safetensor shard 内部 header 的 tensor key，并同步重写
`model.safetensors.index.json`；只改 index 不够。tensor payload 可以 byte-preserving，
但仍需重写约 160 GB shard，带来额外磁盘、I/O 和一次新的完整性验证。因此更推荐
在 vLLM 增加双 namespace mapper。

### 推荐的最小 vLLM 实施项

按优先级建议：

1. 使用最新 `origin/main`/稳定的 `v0.27.1+` 构建匹配 wheel；旧 0.24 wheel 不作为
   DSV4 验证环境；
2. `DeepseekV4Config` 同时兼容 `layer_types+compress_rates`、嵌套
   `rope_parameters.main/compress`、`mlp_layer_types=hash_moe`；
3. NVIDIA `DeepseekV4ForCausalLM` 增加与 AMD 相同的
   `packed_modules_mapping`；
4. 增加完整 HF→native `WeightsMapper`，同时作用于 weight key 与 AutoGPTQ
   module metadata；
5. AutoGPTQ metadata 只由 `qweight` 判定量化 module，排除三个 `tid2eid`；
6. 在 quant config 中增加 `.*\.ffn\.experts$ -> W4/G64`；
7. NVIDIA `_o_proj` 增加 inverse-RoPE + BF16 grouped einsum fallback；
8. 先做 config/meta construct、单层全参数 load 和单层数值对比，再做 4 卡 8K
   demo，避免一开始加载 160 GB 后才发现结构错误。

在 RTX PRO 6000（SM120）上完成上述适配后，启动参数至少应包含：

```bash
vllm serve /path/to/dsv4-gptq-vllm \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --quantization gptq \
  --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --max-model-len 8192 \
  --trust-remote-code
```

不要启用 `deep_gemm_mega_moe`：该路径面向官方 MXFP4 expert，而当前 checkpoint
应走 AutoGPTQ/Marlin W4 MoE。A100/SM80 也不能作为当前 DSV4 sparse-attention
端到端验证机器；现有 DSV4 CUDA backend 目标是 Hopper/Blackwell 路径，SM120
服务器更合适。

### baicai W4A16 为什么更接近 vLLM 当前假设

`baicai1145/DeepSeek-V4-Flash-0731-W4A16` 与当前 AutoRound checkpoint 不是同一种
格式：

| 项目 | baicai compressed-tensors | 当前 AutoRound GPTQ |
|---|---|---|
| namespace/config | 官方 native，含 `compress_ratios`、`num_hash_layers`、flat RoPE config | HF namespace，新 Transformers schema |
| attention | FP8 block `[128,128]`，包含带 scale 的 `wo_a` | q/kv/`wo_b` W8/G64，`wo_a` BF16 |
| routed expert | packed INT4、group 32 | GPTQ INT4、group 64 |
| shared expert/compressor/indexer | BF16 | shared expert W8，compressor/indexer BF16 |
| fused targets | quant config 显式包含 `fused_wqa_wkv` 等 native target | dynamic 仍是逐 HF expert projection regex |

baicai 因此能沿用 vLLM 当前 FP8 `wo_a` 和 native loader 设计。它适合作为 SM120
上的 vLLM DSV4 baseline，但不能证明我们的 GPTQ checkpoint 也可直接加载。

本地 vLLM 0.24 的 compressed-tensors `o_proj` 兼容还不完整；支持
`weight_scale`/`weight_scale_inv` fallback 的修复已在最新 `origin/main`，tag 从
`v0.26.1rc0`/`v0.27.0` 系列开始包含。实际复现建议直接使用匹配构建的
`v0.27.1+`，而不是只按 README 的宽泛“vLLM >= 0.26”判断。

## vLLM 隔离 worktree 最小兼容补丁结果（2026-08-13）

补丁已经只落在隔离 worktree：

```text
/data01/home/yujy/work/vllm_dsv4_gptq
```

原目录 `/data01/home/yujy/work/vllm` 未修改，checkpoint 的 38 个 safetensor shard
也未重写。实现采用“运行时双 namespace mapper + 原 GPTQ payload”的路线，包含：

1. AutoGPTQ 只以 `.qweight` 判断量化 Linear，`tid2eid` 不再误计为量化模块；
2. mapper 覆盖 HF attention、MoE、mHC、Compressor/Indexer 到 vLLM native/internal
   namespace，并增加 NVIDIA `packed_modules_mapping`；
3. AutoGPTQ 在映射前解析每个具体 module 的 dynamic override，映射后保留结果；
   fused routed MoE 会检查其所有子 expert/projection 的 scheme 完全一致，再采用
   W4/G64，因而不需要硬编码一个 DSV4 专用 dynamic regex；
4. config loader 从 HF schema 推导 `num_hash_layers=3` 和
   `compress_ratios=[1,1,4,128,...]`，并按 attention layer 选择嵌套的
   `rope_parameters.main/compress`，且不再原地污染共享 config；
5. NVIDIA `o_proj` 在 `wo_a` 没有 `weight_scale/weight_scale_inv` 时走 inverse
   GPT-J RoPE + BF16 grouped einsum，再进入 W8 `wo_b`；已有 FP8 checkpoint 仍走
   原 DeepGEMM FP8 分支。

### CPU/meta 和真实 tensor loader 证据

使用当前 repaired checkpoint 原目录、原 HF key，未做脚本级 config/key monkey
patch，完成了完整 43 层 CPU/meta 构造：

```text
quant modules                         33,325
fused_wqa_wkv                         AutoGPTQ W8/G64
wo_b                                  AutoGPTQ W8/G64
shared gate_up/down                   AutoGPTQ W8/G64
routed experts                        AutoGPTQ W4/G64
wo_a                                  unquantized BF16
routed w13_qweight shape              [256, 512, 4096]
routed w2_qweight shape               [256, 256, 4096]
```

全 index 仅做名称映射审计：134,198/134,198 个 tensor 均得到 mapped key，映射后
残留 `.self_attn/.mlp/.attn_hc/.ffn_hc/.hc_head.` 旧 namespace 的数量为 0。

代表 tensor 使用 safetensors 中的真实 CPU payload（不是只造 shape）执行 meta
loader，以下 14 项全部 `LOAD_ONE_OK`：

- W8 attention `q_a_proj` 的 `qweight/qzeros/scales/g_idx`；
- W4 routed expert `gate_proj` 的 `qweight/qzeros/scales/g_idx`；
- W8 shared expert `gate_proj.qweight` 和 `o_b_proj.qweight`；
- BF16 `o_a_proj.weight`；
- 非量化 hash router `tid2eid`；
- BF16 Indexer compressor `kv_proj.weight` 与 `position_bias`。

输出记录在：

```text
/tmp/dsv4_vllm_source_patch_meta.out
```

单元测试结果：

```text
tests/quantization/test_auto_gptq.py（新增目标用例）  2 passed
tests/models/test_deepseek_v4_gptq.py                 4 passed
ruff check                                            passed
```

BF16 `wo_a` 单测逐元素对照显式 inverse-RoPE + grouped einsum reference。这里完成的
是 CPU 数学/结构验证；真正的 SM120 kernel、整层输出和 8K generation 仍必须在
RTX PRO 6000 上验证，不能用 meta 构造替代。

### SM120 构建与运行准备

应从隔离 worktree 构建/安装 vLLM，并直接指向现有 repaired checkpoint，不需要
复制 150+ GB 权重：

```bash
cd /data01/home/yujy/work/vllm_dsv4_gptq

vllm serve \
  /data01/home/yujy/work/gptqmodel/work_dirs/deepseek-v4-flash-full43-AutoRoundV2-W8E4G64-smoke-n1-s16-i1-nooffload-repacked-clamped-v2 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --quantization gptq \
  --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --max-model-len 8192 \
  --trust-remote-code
```

GPU 验证的最低验收顺序是：模型完整 load → 8K prefill → decode 512 token → 与
GPTQModel 同 prompt 比较非重复输出、首 token/logits、逐层 hidden state，并单独
确认 mHC、SWA、CSA、HCA 四条路径。当前补丁没有在 A100 上启动，因为该机器不能
代表 SM120 DSV4 sparse-attention backend。
