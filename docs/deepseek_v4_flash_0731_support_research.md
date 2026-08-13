# DeepSeek-V4-Flash-0731 支持与 MXFP4→MXINT8-B64 调研

> 更新说明（2026-08-11）：当前部署目标已经切换为
> `symmetric W4G64 + FP16 scale`，并开始 XH2 静态图实施。本文保留
> MXINT8-B64 的历史扫描与推导；当前实现状态、mHC/CSA/HCA、独立 K/V
> Cache 和 W4G64 结论见
> [DeepSeek-V4-Flash-0731：W4G64 量化与 XH2 静态图适配设计](deepseek_v4_flash_w4g64_xh2_adaptation.md)。

> - 文档状态：调研结论，尚未进入代码实施
> - 调研日期：2026-08-10
> - 目标模型：`/data01/datasets/DeepSeek-V4-Flash-0731`
> - 当前维护仓库：`/data01/home/yujy/work/gptqmodel`
> - 上游参考仓库：`/data01/home/yujy/work/office/GPTQModel`

## 1. 摘要

本次调研覆盖三个问题：

1. 上游 GPTQModel 对 DeepSeek-V4 的支持已经达到什么状态。
2. Hugging Face、ModelScope 上公开的 DeepSeek-V4-Flash INT4/FP4 权重效果如何，与官方原生 mixed MXFP4/FP8 checkpoint 存在多大差距。
3. 官方 routed expert 使用 MXFP4 block32，而目标芯片使用 MXINT8 block64 时，两个 block32 合并成一个 block64 是否必然有损，以及是否需要重排。

核心结论如下：

- 上游 GPTQModel 已具备 DeepSeek-V4 的模型结构识别、量化模块枚举，以及原生 FP4 routed-expert 解码能力，但没有完整验证 `DeepSeek-V4-Flash-0731 + DSpark/MTP`。
- 公开的精确 0731 INT4 checkpoint 缺少可信的端到端质量评测；目前只有 Unsloth 的动态混合 GGUF 提供较完整的 PPL/KL 数据，但它不能直接代表普通 INT4 或目标芯片格式。
- 对本地 0731 全部 routed-expert 权重进行全量扫描后，`MXFP4-B32 → MXINT8-B64` 几乎可以直接无损完成：4,630,511,616 个 block64 中只有 270 个不能严格无损，block 无损率为 `99.999994169%`。
- 因此首版方案不应做全局重排，而应采用“无损 fast path + 极少数异常块最优 exponent 或稀疏修正”。
- 不建议把 FP4 先整体解码成 BF16 再量化；应直接从 packed E2M1 FP4 和 E8M0 scale 转成目标 MXINT8-B64。

## 2. 术语与范围

### 2.1 原生 checkpoint

本文所说的“官方原生”是 DeepSeek 发布的 mixed checkpoint：

- routed MoE experts：MXFP4，E2M1 element，E8M0 shared scale，block size 32；
- attention、shared experts 及其他大部分参数：FP8/BF16 等格式；
- checkpoint 并不是全模型 BF16，也不是全模型 MXFP4。

### 2.2 MXINT8-B64

OCP Microscaling Formats v1.0 中，标准 MXFP4 和 MXINT8 的 scaling block size 都是 32。目标芯片使用的 block size 是 64，因此本文将其明确记作 `MXINT8-B64`，表示厂商扩展，避免和标准 MXINT8-B32 混淆。

### 2.3 分析范围

共享指数合并和转换分析仅针对原生 routed-expert FP4 权重，包括主模型 experts 和 checkpoint 内的 MTP experts。attention、shared experts 等原生 FP8 张量属于另一条转换路径，不包含在本文的 MXFP4→MXINT8 误差统计中。

## 3. 上游 GPTQModel 支持状态

### 3.1 仓库版本差异

| 项目 | 当前维护仓库 | 上游参考仓库 |
|---|---:|---:|
| 分支 | `develop` | `main` |
| GPTQModel 版本 | `5.8.0` | `7.3.4` |
| Transformers 最低版本 | `>=4.57.1` | `>=5.4` |
| safetensors 最低版本 | `>=0.6.2` | `>=0.7` |
| defuser 最低版本 | `>=0.0.6` | `>=0.0.25` |
| `deepseek_v4` 模型定义 | 无 | 有 |

当前维护仓库的 `deepseek_v3.py`、模型结构解析、checkpoint 加载和 defuser 流程包含较多本地扩展，与上游当前实现已经明显分叉。因此不适合直接 cherry-pick 上游完整提交，建议选择性移植。

### 3.2 上游主要支持提交

上游支持由两笔核心提交构成：

1. [`ed42bc58`：support deepseek_v4](https://github.com/ModelCloud/GPTQModel/commit/ed42bc582c9a0aceaca45f09bb8f4e2cb6af483d)
   - 增加 `deepseek_v4` auto mapping；
   - 新增 `DeepSeekV4QModel`；
   - 扩展模型结构解析；
   - 增加模型和结构测试。
2. [`7b277073`：DeepSeek-V4-Pro expert dequant BF16](https://github.com/ModelCloud/GPTQModel/commit/7b277073a773bb3d5276d80ca5ccdb56ad56f0bf)
   - 增加 packed FP4 E2M1 解码表；
   - 读取 per-32 E8M0 scale；
   - 将 routed expert 原生 FP4 解码为 BF16；
   - 增加合成张量解码测试。

主要代码位置：

- `gptqmodel/models/definitions/deepseek_v4.py`
- `gptqmodel/utils/model_dequant.py`
- `tests/test_deepseek_v4_support.py`
- `tests/models/test_deepseek_v4.py`

### 3.3 模型定义覆盖范围

上游 `DeepSeekV4QModel` 继承自 `DeepSeekV3QModel`，主要 module tree 如下：

| 模块 | 处理方式 |
|---|---|
| `q_a_norm`、`q_b_norm`、`kv_norm` | 排除量化 |
| `q_a_proj`、`q_b_proj` | attention quant group 0 |
| `o_a_proj` | 排除量化 |
| `o_b_proj` | attention quant group 1 |
| `kv_proj` | attention quant group 2 |
| MoE router `gate` | 排除量化 |
| routed expert `gate_proj/up_proj` | expert quant group 0 |
| routed expert `down_proj` | expert quant group 1 |
| shared expert `gate/up/down` | 被模型定义枚举 |

模型定义使用：

```python
dynamic_expert_index = "n_routed_experts"
rotary_embedding = "model.rotary_emb"
```

### 3.4 原生 FP4 解码状态

上游 FP4 解码采用 E2M1 value table：

```text
0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

其 checkpoint 假设与本地 0731 一致：

- 两个 4-bit code 打包在一个 `I8` 中；
- 每 32 个逻辑权重共享一个 `F8_E8M0` scale；
- 只转换名称中包含 `.experts.` 且不包含 `.shared_experts.` 的 routed expert 张量；
- 目标输出为 BF16，而不是目标芯片 MXINT8-B64。

这条路径证明上游已经理解官方 checkpoint 的 E2M1/E8M0 存储布局，但它不是面向目标芯片的直接转换实现。

### 3.5 测试覆盖不足

上游当前测试不能证明 0731 已经完整可用：

- `tests/test_deepseek_v4_support.py` 只有 auto class selection 和 module list expansion 两类小型单测；
- `tests/models/test_deepseek_v4.py` 使用 BF16 模型路径，而不是官方 mixed MXFP4/FP8 checkpoint；
- 快速兼容模式默认只量化前两层，然后在混合模型上跑 ARC Challenge；
- FP4 dequant 测试使用合成小张量，没有加载完整约 167GB checkpoint；
- 没有覆盖 `DeepSeek-V4-Flash-0731` 名称；
- 没有覆盖 DSpark、`mtp.0/1/2` 或 speculative decoding acceptance；
- 没有覆盖完整 43 层 W4 质量和端到端 agent benchmark。

此外，上游最初的 V4 支持提交早于 0731 release。官方 0731 模型卡明确说明该版本与 DeepSeek-V4-Flash-DSpark 结构一致，并附带 speculative decoding 模块。因此，上游当前状态应定义为：

> V4 主干结构和 routed-expert FP4 存储格式已经支持；0731 checkpoint、MTP/DSpark 及端到端质量尚未被完整验证。

## 4. 本地 DeepSeek-V4-Flash-0731 权重结构

### 4.1 基本配置

对 `/data01/datasets/DeepSeek-V4-Flash-0731` 进行只读检查后，得到：

| 项目 | 数值 |
|---|---:|
| safetensors shards | 48 |
| checkpoint 总字节数 | 166,878,536,440 bytes |
| decoder layers | 43 |
| hidden size | 4096 |
| routed experts | 256 |
| shared experts | 1 |
| top-k experts | 6 |
| `expert_dtype` | `fp4` |
| 全局非 expert 量化 | FP8 E4M3，block `[128,128]` |

### 4.2 dtype 统计

| dtype | 元素数/字节数 | 说明 |
|---|---:|---|
| `I8` | 148,176,371,712 | packed routed FP4，每字节两个逻辑权重 |
| `F8_E8M0` | 9,261,408,000 | 绝大部分是 routed FP4 scale |
| `F8_E4M3` | 6,304,038,912 | attention 等 FP8 权重 |
| `BF16` | 1,483,567,488 elements | norm、embedding 等 |
| `F32` | 37,741,630 elements | 辅助参数 |
| `I64` | 2,327,040 elements | 索引等 |

其中 routed experts 对应：

```text
packed FP4 bytes       = 148,176,371,712
logical FP4 weights    = 296,352,743,424
routed E8M0 scales     =   9,261,023,232
logical weights / 32   =   9,261,023,232
```

### 4.3 典型 expert 张量

| 权重 | packed shape | 逻辑 shape | scale shape |
|---|---|---|---|
| `w1/gate_proj` | `[2048, 2048]` | `[2048, 4096]` | `[2048, 128]` |
| `w3/up_proj` | `[2048, 2048]` | `[2048, 4096]` | `[2048, 128]` |
| `w2/down_proj` | `[4096, 1024]` | `[4096, 2048]` | `[4096, 64]` |

scale 的最后一维恰好等于逻辑 K 维除以 32。

### 4.4 DSpark/MTP 风险

checkpoint 中存在以下不一致：

- `config.json`：`num_nextn_predict_layers = 1`；
- 实际 checkpoint：包含异构的 `mtp.0`、`mtp.1`、`mtp.2`；
- inference config：`n_mtp_layers = 3`；
- DSpark target layer 为 40、41、42。

因此模型加载和权重转换不能只依赖 `num_nextn_predict_layers` 枚举 MTP。适配时应直接基于 checkpoint tensor names 或 DSpark inference config 识别三个 stage。

## 5. Hugging Face / ModelScope 公开量化调研

### 5.1 精确 0731 checkpoint

| 模型 | 方案 | 公开质量证据 | 结论 |
|---|---|---|---|
| [baicai1145/DeepSeek-V4-Flash-0731-W4A16](https://huggingface.co/baicai1145/DeepSeek-V4-Flash-0731-W4A16) | routed experts：symmetric INT4 group32，BF16 scale；attention 保持 FP8 | tensor-level SNR 约 25.8dB；未公布 PPL/KL/任务表 | 格式最接近目标对照，但模型级质量尚未证明 |
| [ModelScope mirror](https://modelscope.cn/models/baicai1145/DeepSeek-V4-Flash-0731-W4A16) | 同上 | 未发现独立评测 | ModelScope 主要是同 artifact 镜像 |
| [bullerwins/DeepSeek-V4-Flash-0731-AutoRound-W4A16](https://huggingface.co/bullerwins/DeepSeek-V4-Flash-0731-AutoRound-W4A16) | INT4 group128，AutoGPTQ layout | `iters=0`、`model_free=true`，无质量表 | 实质接近 RTN，group128 风险高于 group32 |
| [ModelScope mirror](https://modelscope.cn/models/bullerwins/DeepSeek-V4-Flash-0731-AutoRound-W4A16) | 同上 | 未发现独立评测 | 不能据模型名判断为经过充分 AutoRound 优化 |
| [Unsloth DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | 动态 mixed GGUF，包含 IQ/Q_K 等方案 | 提供 KLD、PPL、top-1 agreement | 数据最完整，但不是普通 uniform INT4/MXINT8 |

Baicai checkpoint 的转换策略与本次目标最接近：它只把原生 routed expert FP4 转成 symmetric INT4 group32，attention 等部分保持官方格式。其模型卡只提供约 25.8dB 的张量 SNR，尚不足以判断 agent/coding 能力是否保持。

Bullerwins checkpoint 虽然名称中包含 AutoRound，但配置为 `iters=0`、`model_free=true`，没有进行典型的迭代校准优化，不能作为高质量 AutoRound 结果的证据。

### 5.2 Unsloth 0731 数值结果

Unsloth 作者以 `UD-Q8_K_XL` 作为自身 lossless baseline，公开数据如下：

| Quant | Size | Mean KLD | PPL | Top-1 agreement |
|---|---:|---:|---:|---:|
| `UD-Q8_K_XL` | 161.9GB | 0 | 5.3322 | 100.00% |
| `UD-Q4_K_XL` | 155.1GB | 0.01324 | 5.3380 | 96.04% |
| `UD-IQ4_XS/NL` | 137.9GB | 0.11501 | 5.5503 | 88.78% |

相对其 Q8 baseline：

- `UD-Q4_K_XL` PPL 增加约 `0.109%`；
- `UD-IQ4_XS/NL` PPL 增加约 `4.09%`。

这组结果说明：保留高敏感张量、采用动态混合格式时，4-bit 可以接近原始 mixed checkpoint；但 155GB 的 `Q4_K_XL` 并不是纯 INT4，不能直接外推到 MXINT8-B64。

完整数据见：[Unsloth KLD table](https://www.reddit.com/r/unsloth/comments/1vc7i2e/q8_k_xl_vs_mxfp4_naming_for_deepseek_v4_flash_0731/)。该表为模型发布者自测结果，不是独立第三方 benchmark。

### 5.3 Preview/Base 模型旁证

#### NVIDIA NVFP4 Preview

[NVIDIA DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) 基于 Preview 而非 0731，并采用 NVFP4 W4A4。其任务级结果为：

| Benchmark | Baseline | NVFP4 | 差值 |
|---|---:|---:|---:|
| GPQA Diamond | 0.894 | 0.891 | -0.003 |
| AA-LCR | 0.658 | 0.655 | -0.003 |
| τ²-Bench Telecom | 0.943 | 0.942 | -0.001 |
| SciCode | 0.481 | 0.481 | 0 |
| IFBench | 0.788 | 0.795 | +0.007 |

它可以证明经过校准的 4-bit 在 V4 Preview 上能够非常接近 baseline，但不能直接证明 0731 INT4 或 MXINT8-B64 的质量。

#### 其他格式转换先例

- [MJPansa/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4)：只转换 routed experts，将 group32 scale 复制为两个 group16 scale，FP4 code 保持 byte-identical；属于格式重容器，不是 INT4 质量对照。
- [Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)：强调 bit-exact recontainer，并保留三个 MTP stage；未公布任务级质量。
- [Intel/DeepSeek-V4-Flash-W4A16-AutoRound](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)：Preview 模型，INT4 group128、RTN/model-free，无公开质量表。
- [EnsueAI/DeepSeek-V4-Flash-Base-INT4](https://huggingface.co/EnsueAI/DeepSeek-V4-Flash-Base-INT4)：Base 而非 0731 instruct；自报 PPL ratio 1.020、KL mean 0.064、agreement 0.922，只能作为量化方法旁证。

### 5.4 公开效果结论

截至本次调研，没有找到满足以下全部条件的公开结果：

1. 精确使用 `DeepSeek-V4-Flash-0731`；
2. 普通 INT4 或目标 MXINT8-B64，而非动态 mixed GGUF/NVFP4；
3. 与官方原生 mixed MXFP4/FP8 checkpoint 使用同一 prompt、encoding、推理参数；
4. 提供完整 PPL/KL 和 agent/coding benchmark。

因此公开数据只能给出边界：

- 高质量动态 mixed 4-bit 可以做到约 `0.1%` 的 PPL 增幅；
- 更激进的 4-bit 方案可能出现约 `4%` PPL 增幅和明显 top-1 agreement 下降；
- 普通 W4A16 group32/group128 对 0731 的真实 agent 能力差距仍需内部实测。

## 6. MXFP4-B32 → MXINT8-B64 数学推导

### 6.1 单个 block32 的无损 INT5 映射

源 MXFP4 权重可以写成：

```text
x = m4 × 2^E
```

其中：

```text
m4 ∈ {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

定义：

```text
q5 = 2 × m4
```

则：

```text
q5 ∈ {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}
x  = q5 × 2^(E-1)
```

因此，对于每个原始 block32：

- element 从 FP4 E2M1 转成 signed INT5 code；
- shared exponent 从 `E` 变为 `E-1`；
- 整个过程严格无损。

### 6.2 两个 block32 合并

设相邻两个 block32 的源 shared exponent 为 `E0`、`E1`，定义：

```text
Et = min(E0, E1) - 1
Δ  = abs(E0 - E1)
```

使用 `Et` 作为目标 block64 的公共 exponent：

- 较小 exponent 半边保持 `q5`；
- 较大 exponent 半边将 `q5` 左移 `Δ` 位；
- 最大绝对 code 为 `12 × 2^Δ`。

| Δ | 最大绝对 INT code | signed INT8 是否保证容纳 |
|---:|---:|---|
| 0 | 12 | 是 |
| 1 | 24 | 是 |
| 2 | 48 | 是 |
| 3 | 96 | 是 |
| 4 | 192 | 不保证，取决于实际 mantissa |
| 5 | 384 | 不保证，取决于实际 mantissa |

由此得到：

> 任意 `Δ≤3` 的相邻 MXFP4 block32 都可以严格无损地合并为一个 MXINT8-B64 block。

`Δ>3` 并不等价于一定有损。如果较大 exponent 半边没有使用大幅值 FP4 code，左移后仍可能落在 INT8 范围内。

## 7. 共享指数全量扫描

### 7.1 扫描范围和方法

扫描范围包括：

- 43 层主模型的所有 routed expert `w1/w2/w3`；
- checkpoint 中三个 MTP stage 的 routed expert；
- 共 35,328 个 scale tensors；
- 9,261,023,232 个 E8M0 scale；
- 4,630,511,616 对相邻 block32，即 4,630,511,616 个目标 block64。

第一遍读取所有 E8M0 scale，统计 exponent 和相邻 exponent pair。第二遍只读取 `Δ>3` 的 block 对应 FP4 mantissa，检查实际 INT8 可表示性并枚举目标 exponent。

### 7.2 共享指数集合

所有 routed-expert shared exponent 的解码值只有：

```text
{-9, -8, -7, -6, -5, -4, -3, -2, -1}
```

对应分布：

| E | scale | 数量 | 占比 |
|---:|---:|---:|---:|
| -9 | `2^-9` | 814 | 0.000008790% |
| -8 | `2^-8` | 848,704 | 0.009164257% |
| -7 | `2^-7` | 3,488,897,271 | 37.672913496% |
| -6 | `2^-6` | 5,180,504,355 | 55.938790188% |
| -5 | `2^-5` | 557,595,293 | 6.020882132% |
| -4 | `2^-4` | 33,119,724 | 0.357624888% |
| -3 | `2^-3` | 52,776 | 0.000569872% |
| -2 | `2^-2` | 4,163 | 0.000044952% |
| -1 | `2^-1` | 132 | 0.000001425% |

`E=-7/-6/-5` 合计超过 99.6%，整个模型的 shared exponent 高度集中。

### 7.3 最常见相邻 exponent pair

| `(E0,E1)` | 占全部 block64 比例 |
|---|---:|
| `(-6,-6)` | 34.146921% |
| `(-7,-6)` | 21.500912% |
| `(-6,-7)` | 21.497285% |
| `(-7,-7)` | 16.147579% |
| `(-5,-5)` | 5.407903% |

这五类合计约 98.7%，且全部满足 `Δ≤1`，可以严格无损合并。

扫描中共出现 62 种 ordered exponent pair。

### 7.4 exponent 差值统计

| 区域 | block64 数量 | mean Δ | `Δ>1` | `Δ>2` | `Δ>3` | 最大 Δ |
|---|---:|---:|---:|---:|---:|---:|
| 主模型 | 4,328,521,728 | 0.46264 | 0.0526666% | 0.0002171% | 697 | 5 |
| MTP | 301,989,888 | 0.16476 | 极少 | 极少 | 36 | 4 |
| 合计 | 4,630,511,616 | — | — | — | 733 | 5 |

`Δ>3` 的比例约为 `1.58×10^-7`，已经非常稀少。

### 7.5 检查实际 FP4 mantissa 后的结果

对 733 个 `Δ>3` block64 读取实际 64 个 FP4 code 后：

| 结果 | block 数量 |
|---|---:|
| `Δ>3` 但实际仍严格可表示 | 463 |
| 无法用单一 MXINT8-B64 exponent 严格表示 | 270 |

全模型最终结果：

```text
exact blocks      = 4,630,511,346
lossy blocks      =           270
exact block rate  = 99.999994169111%
lossy block rate  = 5.830889162810e-08
```

在 733 个 exception block 上枚举公共 exponent 并选择最小 MSE 后，约 98.4673% 的 exception elements 仍保持精确。折算到全部 296,352,743,424 个 routed-expert 逻辑权重，约只有 719 个值发生变化：

```text
estimated changed weight rate = 2.426162794016e-09
estimated changed percentage  = 0.000000242616%
```

### 7.6 exception exponent pair

| exponent pair | 数量 |
|---|---:|
| `(-7,-3)` | 357 |
| `(-3,-7)` | 208 |
| `(-6,-2)` | 66 |
| `(-2,-6)` | 49 |
| `(-7,-2)` | 15 |
| `(-8,-4)` | 11 |
| `(-5,-1)` | 8 |
| `(-2,-7)` | 6 |
| `(-4,-8)` | 4 |
| `(-1,-5)` | 4 |
| `(-8,-3)` | 3 |
| `(-3,-8)` | 2 |

### 7.7 exception 集中位置

较多 exception 的位置包括：

| 位置 | `Δ>3` block 数量 |
|---|---:|
| layer 42 `w2` | 190 |
| MTP `w2` | 36 |
| layer 17 `w3` | 35 |
| layer 20 `w2` | 31 |
| layer 19 `w3` | 30 |
| layer 22 `w2` | 26 |
| layer 41 `w2` | 26 |

这表明异常不是均匀分布，而是集中在少数 layer/projection。转换器应输出按 layer、expert 和 projection 分类的统计，便于后续定向验证。

## 8. 异常块量化策略

### 8.1 推荐的默认策略

首版转换器不做重排，按以下流程处理每个 block64：

1. 解包两个 source block32 的 E2M1 FP4 code。
2. 映射成无损 `q5=2×m4`。
3. 若 `Δ≤3`，使用：

   ```text
   Et = min(E0,E1) - 1
   ```

   并对较大 exponent 半边左移 `Δ` 位。
4. 若 `Δ>3`，先检查实际最大 code 是否仍可在该 `Et` 下无损表示。
5. 若不能无损表示，在小范围候选 exponent 中进行枚举，选择 MSE 最低的 exponent，执行 round-to-nearest-even 和 clamp。

由于全模型只有 270 个真正 lossy block，仅使用 weight MSE 已可能足够。若后续端到端验证发现这些位置敏感，再加入 activation variance 或 GPTQ Hessian 加权误差。

### 8.2 exception 策略对比

仅在 exception block 上的初步误差趋势为：

| 策略 | exception-block relative MSE |
|---|---:|
| 固定使用较小 scale | 0.0444 |
| 固定使用较大 scale | 0.0244 |
| range-safe exponent | 0.0002103 |
| 逐块枚举 MSE 最优 exponent | 0.000046886 |

因此，即使不做稀疏修正，也不应简单固定选择较小或较大 source scale；至少应使用 range-safe 或逐块候选枚举。

### 8.3 bit-exact 要求下的策略

如果芯片验收要求相对原始 MXFP4 完全 bit-exact，可以只对 270 个 block 增加异常 sidecar，例如：

- block id；
- 第二个 source exponent；
- 溢出元素的稀疏 residual；
- 或将该 block 拆成两个带 padding 的独立计算单元。

主 MXINT8 GEMM 完成后，再进行一次极小的 sparse residual accumulation。其存储开销可以忽略，但需要评估 kernel 调度和稀疏修正是否值得。

## 9. 是否需要重排

### 9.1 结论

不建议为当前模型执行全局 channel reorder。270 个 lossy block 相对于 4.63B 个 block 的比例过低，重排引入的图变换、activation layout 和 kernel 复杂度远高于收益。

### 9.2 每行独立排序不可直接用于 GEMM

在 scale sample 上，对每个输出行独立按 exponent 排序，可以将 mean Δ 从约 0.45 降低到约 0.009～0.018，并消除 `Δ>3`。

但不同输出行会得到不同的 K permutation。这等价于每个输出 channel 使用不同的 activation 排列，普通矩阵乘法无法共享同一个输入向量，因此只能视为不可实现的 oracle 上界。

### 9.3 全输出行共享的 K permutation 收益很小

使用所有输出行共享的 block permutation，例如按每个 K block 的平均 exponent 排序，在 sample 上 mean Δ 仅从约 0.448 降到约 0.447。原因是 scale 分布随输出行变化明显，单一全局 K permutation 无法同时优化所有行。

这类 permutation 还会影响上游 activation、residual、attention 和其他线性层，不值得为极少数 exception 引入。

### 9.4 MoE FFN 中可严格等价的重排

MoE FFN 可表示为：

```text
y = W2(SiLU(W1x) ⊙ W3x)
```

对 intermediate channel 使用 permutation matrix `P`：

```text
W1' = P W1
W3' = P W3
W2' = W2 P^-1
```

则网络输出严格不变。

这意味着：

- 可以同步重排 `w1/w3` 的输出行和 `w2` 的输入列；
- 能改变 `w2` 的 K 维 block 配对；
- 不能改善 `w1/w3` 的 K 维 block 配对，因为对 `w1/w3` 做的是 N 维重排；
- `w2` 输出位于 residual hidden space，不能任意重排输出行，除非在残差相加前恢复。

如果未来硬件不允许异常 sidecar、又强制零误差，可以只针对 `w2` 研究这类严格等价的 intermediate-channel permutation。但当前统计下不应作为首选方案。

### 9.5 硬件二维 tile 的可能性

如果芯片所谓 block64 不要求 64 个元素都沿 K 连续，而允许将一个 scale 作用于 `2×32` 或其他 N×K tile，则可以考虑跨输出行配对两个 block32，并结合输出 channel permutation 降低 exponent 差。

该方案是否成立取决于硬件：

- scale 的广播维度；
- weight DMA/layout；
- accumulation 语义；
- kernel 是否允许二维 block。

在获得明确硬件 layout 规格前，不应假设这条路径可用。

## 10. 存储与带宽影响

routed-expert 逻辑权重为 296,352,743,424 个：

| 格式 | 权重 payload，不含其他张量 |
|---|---:|
| packed FP4 | 148.176GB |
| physical INT8 | 296.353GB |
| BF16 | 592.705GB |

此外：

- source block32 有约 9.261GB E8M0 scales；
- target block64 scale 数量减半，约 4.631GB；
- physical INT8 仍会使 routed-expert payload 接近翻倍；
- 先解码 BF16 会造成超过 592GB 的 routed-expert 临时/落盘数据，完整模型更大。

推荐转换链：

```text
packed E2M1 FP4 + E8M0/B32
        ↓ direct conversion
MXINT8/B64 + E8M0/B64
```

不推荐：

```text
packed FP4 → BF16 → 再量化 MXINT8
```

如果落盘和 PCIe 传输是瓶颈，可以考虑：

- 继续保存官方 packed FP4，在加载芯片前在线展开；
- 将无损 q5 保持为 packed INT5 中间格式；
- 按 shard 流式转换，避免同时持有 BF16 或完整 INT8 副本。

## 11. 当前仓库适配建议

### 11.1 阶段一：模型结构和加载

选择性移植：

- `DeepSeekV4QModel` 定义；
- `model_type=deepseek_v4` auto mapping；
- V4 attention/MoE module tree；
- 必要的 Transformers V4 model compatibility；
- checkpoint 内 `mtp.0/1/2` 的识别和保留。

不建议直接覆盖当前维护仓库的 `deepseek_v3.py`、`structure.py` 或完整 loader。

### 11.2 阶段二：直接转换器

实现独立转换组件：

```text
input:
  packed I8 expert weight
  F8_E8M0 scale/B32

output:
  signed INT8 weight
  E8M0 scale/B64
  optional exception metadata
```

转换器至少输出：

- tensor/layer/expert/projection；
- source exponent histogram；
- exponent-pair histogram；
- exact block count/rate；
- saturation count；
- changed element count；
- weight MSE/relative MSE；
- exception block index。

### 11.3 阶段三：芯片集成

需要确认：

- signed INT8 是否按 `[-127,127]` 对称使用，还是允许 `-128`；
- block64 是否严格沿 K；
- shared exponent 编码、bias 和合法范围；
- scale 是否可直接使用 E8M0；
- weight layout 和 nibble/endian 顺序；
- accumulator 位宽；
- activation 是 MXINT8、普通 INT8 还是其他格式；
- MTP experts 是否与主模型共用同一 kernel/layout。

### 11.4 阶段四：质量验证

建议建立四组模型：

| 版本 | 说明 | 目的 |
|---|---|---|
| V0 | 官方 native mixed MXFP4/FP8 | 基准 |
| V1 | MXINT8-B64，无重排，异常块逐块最优 exponent | 首选芯片方案 |
| V2 | MXINT8-B64 + exception sparse correction | bit-exact 上界 |
| V3 | 公开 W4A16 group32/group128 或内部 GPTQ/AutoRound | INT4 对照 |

同时分别测试：

- 仅主模型；
- 主模型 + MTP/DSpark；
- DSpark 关闭和开启；
- 不同 reasoning effort；
- 官方专用 `encoding_dsv4.py`。

## 12. 评测指标

### 12.1 转换级

- exact block ratio；
- exact element ratio；
- saturation count；
- max absolute error；
- MSE、relative MSE、NRMSE；
- cosine similarity；
- 按 layer/expert/projection 的异常分布。

### 12.2 模块级

- expert `w1/w2/w3` 输出误差；
- MoE layer 输出 cosine/NRMSE；
- router top-k 一致性；
- residual 前后误差放大；
- MTP 每个 stage 的 logits 误差。

### 12.3 模型级

- calibration corpus logits KL；
- token top-1 agreement；
- WikiText 或内部语料 PPL；
- 固定 prompts 的 greedy generation 一致性；
- reasoning/coding/agent 内部评测集；
- DSpark acceptance rate、平均 accepted tokens、decode TPS。

官方 0731 agent 分数依赖专用 encoding、reasoning effort、采样参数及尚未完全公开的 DeepSeek Harness，不能直接把不同社区 checkpoint 的公开分数横向相减。内部对照必须固定全部推理参数和 prompt encoding。

## 13. 主要风险

1. **版本兼容风险**：当前维护仓库和上游依赖版本跨度较大，V4 Transformers model 可能需要单独 backport。
2. **DSpark/MTP 漏加载风险**：配置字段与实际三个 MTP stage 不一致。
3. **格式混淆风险**：官方模型是 mixed FP4/FP8，不能把所有张量按 FP4 expert 逻辑处理。
4. **存储膨胀风险**：physical INT8 routed experts 接近 packed FP4 的两倍。
5. **公开评测不足**：W4A16 checkpoint 缺少完整 PPL/KL/agent benchmark。
6. **硬件语义风险**：目标 block64、E8M0 编码、layout、INT8 对称范围尚需与芯片规格对齐。
7. **异常块敏感性风险**：虽然 lossy block 极少，但部分集中在 layer 42、MTP 和少数 `w2/w3`，仍需验证是否影响 DSpark 和最终 logits。

## 14. 推荐决策

建议项目当前采用以下技术决策：

1. 将上游状态定义为“V4 主干结构可参考，0731/DSpark 未完整验证”。
2. 选择性移植模型定义和加载能力，不整笔 cherry-pick。
3. 直接实现 MXFP4-B32→MXINT8-B64 转换，不经过全模型 BF16。
4. 首版不做 channel reorder。
5. `Δ≤3` 和实际可表示 block 走严格无损 fast path。
6. 对仅有的 270 个真正异常 block 使用逐块最优 exponent。
7. 若产品要求 bit-exact，再增加稀疏 exception correction，而不是全局重排。
8. 将 MTP/DSpark 作为首轮支持范围，而不是后续补丁。
9. 使用 V0/V1/V2/V3 固定条件对照，内部实测决定是否需要 GPTQ/AutoRound 或重排。

## 15. 参考资料

- [DeepSeek-V4-Flash-0731 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [DeepSeek-V4-Flash Preview 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [GPTQModel DeepSeek-V4 model definition](https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/models/definitions/deepseek_v4.py)
- [GPTQModel DeepSeek-V4 support commit](https://github.com/ModelCloud/GPTQModel/commit/ed42bc582c9a0aceaca45f09bb8f4e2cb6af483d)
- [GPTQModel DeepSeek-V4 FP4 dequant commit](https://github.com/ModelCloud/GPTQModel/commit/7b277073a773bb3d5276d80ca5ccdb56ad56f0bf)
- [Unsloth DeepSeek-V4-Flash-0731 GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- [Unsloth DeepSeek-V4-Flash-0731 KLD table](https://www.reddit.com/r/unsloth/comments/1vc7i2e/q8_k_xl_vs_mxfp4_naming_for_deepseek_v4_flash_0731/)
- [Baicai DeepSeek-V4-Flash-0731 W4A16](https://huggingface.co/baicai1145/DeepSeek-V4-Flash-0731-W4A16)
- [Bullerwins DeepSeek-V4-Flash-0731 AutoRound W4A16](https://huggingface.co/bullerwins/DeepSeek-V4-Flash-0731-AutoRound-W4A16)
- [NVIDIA DeepSeek-V4-Flash NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
- [MJPansa DeepSeek-V4-Flash-0731 NVFP4](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4)
- [Vontra DeepSeek-V4-Flash-0731 MXFP4 MLX](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)
