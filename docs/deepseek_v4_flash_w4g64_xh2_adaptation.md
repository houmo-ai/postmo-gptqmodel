# DeepSeek-V4-Flash-0731：W8/W4-G64 量化与 XH2 静态图适配设计

> 文档状态：实施中，已完成原生权重解码、真实 no-offload mixed GPTQ 与
> AutoRound smoke、GPTQ/AutoRound mixed-precision recipe、Defuser 验证、
> mHC/CSA/HCA 首六层静态图、运行时 Cache ABI、tiny HMONNX、真实尺寸
> first-6 prefill/decode HMONNX 全链路导出、完整六层 GPTQ/AutoRound 量化产物，
> 两种算法完整 packed 产物的 no-offload HMONNX 导出、Host 全量位置/压缩控制 ABI，
> 以及面向完整 43 层的逐 MoE 低内存导出实现。首六层普通/低内存导出的两张
> ONNX 主文件与两份 external data 已分别在 GPTQ、AutoRound 上完成逐字节
> 一致性验证。完整 43 层 AutoRound 单卡量化已经完成；完整 43 层低内存
> HMONNX 正在执行最终主图优化。mHC 少步替换已经完成真实 43 层、四卡、
> FP16、WikiText-2 对照，当前没有候选满足 FP16 logits/PPL 等价门禁。
>
> 更新时间：2026-08-13
>
> 模型：`/data01/datasets/DeepSeek-V4-Flash-0731`
>
> GPTQModel：`/data01/home/yujy/work/gptqmodel`
>
> XH2 ModelZoo：`/data01/home/yujy/work/xh2modelzoo`

## 1. 当前结论

当前目标已经明确为：

- 权重：非共享 routed expert Linear 使用 W4，其余 GPTQModel 已登记的普通
  quantizable Linear 使用 W8；两者均为 symmetric、group size 64、FP16 scale；
- 算法：同一套 profile 同时支持 GPTQ 和仓库内置 AutoRound（默认 V2）；
- offload：默认关闭，仅在调用方显式 opt-in 时启用；
- 量化设备：默认只暴露一张 80-GiB GPU；GPTQ/AutoRound 都按 decoder layer
  和 MoE module chunk 串行推进，单层实测峰值约 24 GiB，不存在容量上的多卡
  必要。多张可见 GPU 只会成为可选 expert worker pool；
- calibration forward data parallel：默认关闭，避免为 forward replay 在每张
  可见 GPU 复制当前层；
- runtime：固定 prefill graph 长度 256，decode graph 长度 1；
- 最大上下文：256 Ki token，即 262,144 token；
- SWA window：128；
- CSA：ratio 4，Indexer TopK 512；
- HCA：ratio 128；
- MLA：保留模型原本的 latent/shared-KV 与分组低秩输出投影，不展开成普通 MHA；
- Cache：逻辑 K/V 数值相同，但因 XH2 K-SEFP 与 V-SEFP 的 scale 归约轴不同，物理 K/V Cache 必须分开。

已经得到的工程结论：

1. 官方 Defuser 0.0.25 已原生支持 `deepseek_v4`，不需要维护本地注册补丁。
2. 本地 GPTQModel 能完整解码官方 FP8/FP4 mixed checkpoint；GPTQ 与
   AutoRound 共享 W8-base/W4-routed-expert 的动态精度规则。
3. 旧全 W4 首层 pilot 已真实完成 775 个子矩阵，证明 dequant、replay、pack、
   save 链路成立，scale 为 FP16、group size 为 64；该 artifact 已被新 profile
   取代，不能作为最终 mixed-precision 产物。
4. CSA 的 8 行 KV/score carry state 可以用固定形状图准确表达；`offset` 不是
   持久 state，而是 Host 根据绝对 `past_length` 为当前调用重新计算的共享输入。
5. `-65504` 可以作为 score state 的有限哨兵；关键不是让全无效 softmax 产生全 0，而是用 `new_count` 禁止无效输出写入 Cache。
6. CSA Indexer 不能在 256 Ki 上直接物化 `[B,256,64,65536]` 全量 score；必须按 Cache 轴分块并做局部 TopK/全局 TopK merge，或最终融合成专用 kernel。
7. mHC 的 4×4 Sinkhorn 用 FP16 做 20 次迭代数值可用，但不能直接减少迭代。
   真实 43 层 FP16 对照中，即使只从 20 减到 19，1,024 个 WikiText-2 token
   的 PPL 也从 8.304733 变为 8.322607，Top-1 只有 965/1,024 相同；12 步
   log-SOR 与 100 步极限近似差异更大。默认继续保留 20 步。
8. 真实首层 mixed GPTQ 已在物理 GPU 4～7 可见、默认 no-offload/no-forward-DP
   条件下完成并保存；但显存分布仅约 21.9/2.7/2.6/2.6 GiB，证明四卡不是容量
   要求。完整 43 层改为只暴露物理 GPU 4 的单卡命令。
9. 同配置的 AutoRound V2 已先完成 layer-0 smoke，随后完成全部 6 层量化；
   完整产物有 4,608 个 W4 routed-expert Linear 和 42 个 W8 attention/shared-
   expert Linear，各层严格为 768+7。此次仍是 `nsamples=1/seqlen=16/iters=1`
   的链路 smoke，不是质量结果。
10. 首六层固定图已经完整表达 `SWA,SWA,CSA,HCA,CSA,HCA`。位置与有效长度是
    Host/scheduler 状态，不再在图中重建；10 个跨层共享的 Host 输入包含 4 个
    attention mask/validity 和 CSA/HCA 各 3 个 compressor control。最终 ABI 为
    50 个输入、13 个输出、22 次 Cache update；图只回传 12 个真正需要跨调用保存的
    compressor KV/score state，不再回传 offset/new-count。
11. 真实尺寸首六层图已用“六层全部量化”的 AutoRound packed checkpoint
    完成 prefill256/decode1 HMONNX 导出；两张图均为 50 输入、13 输出、
    22 个 `KVcache`、12 个 W4 `MoeBlock`、126 个 W8 `Linear`、50 个 fused
    `RMSNorm` 和 26 个 `Rope`。packed-only
    路径直接消费 GPTQModel 的整数 `qweight + FP16 scale`，不再先常驻 dense
    权重作为桥接。最终整张图的 `Cast/ConstantOfShape/Where/Less/LessOrEqual/
    Greater/GreaterOrEqual` 均为 0；仅保留 12 个模型语义要求的 shared-expert
    FP16 `Clip`。
12. Qwen3.5 式低内存导出已经按完整 MoE 边界接入：主图保留 attention、mHC、
    CSA/HCA 和 Cache，仅把 `MoE(hidden_states,input_ids)` 变成 2 入 1 出的零参数
    placeholder；每次流式加载一个 `model.layers.N.mlp`，同时生成 prefill/decode
    子图后释放。真实六层 AutoRound 产物上，低内存与普通导出的 prefill/decode
    主文件和 external data 四项全部 `cmp` 一致；低内存主进程峰值约 36.6 GiB，
    普通路径约 158.0 GiB，二者 GPU 峰值均约 2.63 GB。
13. 完整六层 GPTQ 产物也已通过同一低内存链路：4,608 个 routed-expert W4、
    42 个 base/shared W8，全部 symmetric G64、FP16 scale。GPTQ 的普通与低内存
    prefill/decode 主文件和 external data 四项逐字节一致；最终图仍为
    5,940/5,928 个节点、50 个 fused RMSNorm、26 个 Rope、12 个合法 Clip 和
    10 个 Host position/control 输入，`Cast/ConstantOfShape` 均为 0。GPTQ 与
    AutoRound 的主图逐字节相同，而 external data 从量化参数区域开始不同，说明
    两种算法共享结构但没有错误复用量化权重。

## 2. 官方 checkpoint 到底是什么格式

### 2.1 不是“整个 attention 都以 FP8 运行”

官方 checkpoint 中，多数 attention Linear 权重和 shared expert 权重是：

- element：FP8 E4M3；
- weight block：128×128；
- scale：UE8M0/E8M0 语义的直接 scale；
- checkpoint scale shape：按 128×128 weight tile 排列。

这只说明权重存储格式。它不自动意味着：

- attention activation 一定是 FP8；
- KV Cache 一定是 FP8；
- compressor、router、mHC 参数都是 FP8。

实际 checkpoint 中，compressor、Indexer 的部分投影/position bias、router、norm、mHC 等还包含 BF16/FP32 参数。官方 inference 代码也明确让 compression 和 mHC 的关键归约在 FP32 中执行。

### 2.2 routed expert

256 个非共享 routed experts 的 `gate/up/down` 权重为：

- element：E2M1 FP4；
- 每字节打包两个 4-bit value；
- scale：E8M0；
- block：沿 K 维每 32 个逻辑元素一个 scale。

E2M1 value table 为：

```text
0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

### 2.3 shared expert、compressor、router

- `shared_experts` 是每层始终执行的一个 SwiGLU expert，不参与 Top-6 路由；
- routed experts 是 256 选 6；
- `compressor` 是 CSA/HCA 的 learned gated pooling 参数，不是通用压缩工具；
- `router` 决定 routed expert id 和权重；前 3 层是 hash routing，后续层是 score routing；
- 第 0～2 层的 `tid2eid` 是 token id 到 6 个 expert id 的静态表。

## 3. routed expert W4G64 + FP16 scale 能否“无损”

### 3.1 FP16 scale 带来的自由度

采用普通 FP16 scale，不再要求 scale 本身必须是 2 的整数次幂。对每个 64-element group，可以求：

```text
q_i = clamp(round(w_i / s), q_min, q_max)
w_hat_i = s * q_i
```

并以 weight MSE 或 GPTQ 的 activation-weighted reconstruction error 选择 `s`。这确实能把 block32 合并造成的误差更多地吸收到 scale 中，比强制 E8M0 shared exponent 灵活。

### 3.2 但不能承诺 bit-exact lossless

普通 symmetric INT4 只有有限的等间距 code。原始 E2M1 FP4 是非均匀 value table，而且相邻两个 block32 可能具有不同的 E8M0 exponent。把两组数放进一个 group64 后，要严格无损必须同时满足：

1. 64 个值全部落在同一个均匀格点 `s·Z` 上；
2. 对应整数 code 不越过 INT4 动态范围；
3. 两个原始 exponent 与 E2M1 mantissa 组合后仍共享同一个精确步长。

这对任意 block 不成立。因此：

> FP16 scale 可以显著降低误差，但不能让一般的 MXFP4-B32 → INT4-G64 转换数学上无损。

ModelScope/Hugging Face 上 `baicai1145/DeepSeek-V4-Flash-0731-W4A16` 所说的“基本无损”，目前依据主要是 tensor SNR，而不是 bit-exact、PPL、KL 或完整任务评测。它可以作为对照 artifact，不能作为“严格无损”的证明。

### 3.3 当前落地方法

当前统一 recipe 使用：

```text
base bits          = 8
routed expert bits = 4
group_size         = 64
sym                = true
desc_act           = false
scale              = FP16 in saved GPTQModel artifact
method             = GPTQ | AutoRound V1/V2（默认 V2）
offload_to_disk    = false（默认）
auto forward DP    = false（默认）
```

这里“其余 W8”的准确边界是 GPTQModel 模型定义已经登记的普通 Linear：

| 模块 | 当前精度 |
|---|---:|
| `mlp.experts.<id>.gate/up/down`（非共享 routed） | W4G64 |
| `mlp.shared_experts.gate/up/down` | W8G64 |
| `self_attn.q_a/q_b/kv/o_b` | W8G64 |
| grouped `self_attn.o_a` | BF16 |
| CSA/HCA compressor、Indexer 投影 | BF16 |
| router、mHC、norm、sink、position bias | BF16/FP32 |

前三层的 Hash-MoE 不改变这张表。`num_hash_layers=3` 只让 router 用
`tid2eid[input_ids]` 选出 6 个 expert ID；router 仍计算 learned score 作为路由
权重，随后仍执行与第 4～42 层相同的 256 个真实 SwiGLU expert。官方
GPTQModel 的 `DeepSeekV4QModel.module_tree` 对所有层统一登记
`mlp.experts.#.{gate_proj,up_proj,down_proj}`，没有为 Hash-MoE 做豁免。完整六层
AutoRound artifact 的物理审计也显示 layer 0、1、2 各有 768 个 expert
`qweight`，因此前三层同样走 W4；`tid2eid` 查表和 router weight 不走 W4。

这不是正则遗漏：`o_a` 是分组投影，不能替换成通用 GPTQ Linear；compression、
Indexer 与路由控制支路又直接影响稀疏选择和共享 Gate softmax，当前先沿用官方
高精度边界。若后续要求它们也 W8，需要单独加入模型定义、逐支路校准并重新做
TopK/attention golden，不能仅扩大一个正则。

先把官方 mixed checkpoint 按 shard 流式解码成 BF16，再做 GPTQ 或
AutoRound。这里经过 BF16 是为了复用 calibration/reconstruction 流程，不是
声称它是最终芯片转换的唯一方案。

最终质量版需要至少比较：

- 原生 mixed checkpoint；
- Baicai W4A16 group32；
- RTN W8-base/W4-expert G64；
- GPTQ W8-base/W4-expert G64；
- AutoRound V1/V2 W8-base/W4-expert G64；
- 如果硬件允许，敏感控制支路保留 BF16 的 mixed profile。

## 4. GPTQModel 当前实施状态

### 4.1 已加入的能力

- `deepseek_v4` auto mapping 和模型定义；
- 官方 FP8 E4M3 + direct scale 解码；
- routed expert packed E2M1 FP4 + E8M0 scale 解码；
- first-N-layer BF16 staging，避免“只跳过量化但仍加载 43 层”；
- W8-base/W4-routed-expert G64 的 GPTQ/AutoRound recipe、CLI、provenance；
- routed expert 正则不会命中 `shared_experts`；shared expert 使用基础 W8；
- disk offload 默认关闭，显式 `--offload-to-disk` 才启用；
- calibration forward data parallel 默认关闭，显式
  `--auto-forward-data-parallel` 才启用；
- 推荐执行方式为 `CUDA_VISIBLE_DEVICES=4 ... --device cuda:0`。即使
  `auto_forward_data_parallel=false`，暴露多卡仍可能让 weight-only MoE worker
  pool 分发不同 expert chunk；这只是可选吞吐优化，不是模型装载要求。当前
  Python 3.12 仍有 GIL，官方日志也明确提示该多卡 MoE 加速是 sub-optimal；
- mHC/RMSNorm mixed-dtype replay 的 BF16 autocast；
- dict/tuple 嵌套 `position_embeddings` 的 device 搬运；
- `o_a_proj` 这类非量化 GroupedLinear 不被误替换成普通 HookedLinear；
- offload shell 与真实 turtle 同时执行官方 Defuser 转换。

这里的“默认不 offload”只作用于 DeepSeek-V4 Flash recipe 与对应 XH2
量化入口，没有改变 GPTQModel 其他模型的全局默认。关闭 offload 与关闭
forward data parallel 是两个独立开关：前者控制磁盘迁移，后者控制是否在每张
GPU 上复制完整当前层。量化本体本来就是逐 layer/subset/module chunk 运行，关闭
二者并只暴露一张 GPU 即可完成；一次显式开启 forward DP 的对照试跑在层尾因
复制完整 MoE layer 而 OOM，不应被误解成“需要更多卡”。

XH2 的 routed-expert 打包还必须保留 GPTQModel 反量化 Linear 上的
`quant_weight`。当前实现逐专家复制 dense weight 与 `int8` 容器中的 W4
整数值（有效值域 `[-8,7]`）到 `MoeBlock.expert_*`，随后把源 expert weight
与 `quant_weight` 移到 meta。这样不会在静态图包装时丢失 W4 provenance，
也避免 `torch.stack` 同时额外常驻一套 256-expert 权重。

默认常驻策略已经在三层锁定：

| 入口 | 无显式参数时的结果 |
|---|---|
| GPTQModel DeepSeek-V4 recipe/CLI | `offload_to_disk=false` |
| GPTQModel calibration replay | `auto_forward_data_parallel=false` |
| XH2 quant adapter/CLI | 两个开关均为 `false`，`offload_dir=null` |
| Merak/HMONNX model config | `enable_auto_offload=false` |
| 推荐可见 GPU | 单卡，例如物理 GPU 4 |

GPTQModel 与 XH2 两个真实 CLI 的 dry-run provenance 均已核对；这里不是只依赖
dataclass 默认值或 mock。需要 offload 时仍可显式传
`--offload-to-disk`，不会影响其他模型的全局策略。

### 4.2 Defuser

环境已经升级到：

```text
defuser==0.0.25
```

`requirements.txt` 和 `pyproject.toml` 要求均为 `defuser>=0.0.25`。官方 registry 已包含 `deepseek_v4`，本地没有新增 DeepSeek-V4 registry shim。

offload 验证结果：

| 检查项 | 结果 |
|---|---:|
| shell experts | 256 |
| turtle experts | 256 |
| shell/turtle parameter names | 完全相同 |
| 材料化前首层 meta 参数 | 788 |
| 材料化后首层 meta 参数 | 0 |
| 首个 expert 权重 | 非零 |

### 4.3 旧全 W4 首层端到端 pilot

配置：1 个 calibration sample、sequence length 16、仅量化 layer 0、物理 GPU 4～7。

结果（只证明链路，不是当前放行 profile）：

| 项目 | 数值 |
|---|---:|
| GPTQ 子矩阵 | 775 |
| `qweight` tensors | 775 |
| `qzeros` tensors | 775 |
| `scales` tensors | 775 |
| `g_idx` tensors | 775 |
| scale dtype | FP16 |
| artifact 大小 | 67 GiB |
| 退出状态 | 0 |

775 的构成为：

```text
attention                 4
256 routed experts × 3  768
shared expert × 3          3
合计                     775
```

`o_a_proj` 仍保留原分组低秩权重，没有展开或错误量化。后 5 层在这个
pilot artifact 中保持 BF16。当前 mixed profile 中，同样的 775 个目标拆为：

```text
W4：256 routed experts × 3 = 768
W8：attention 4 + shared expert 3 = 7
```

旧全 W4 的前 6 层中等校准任务已经停止，其不完整 offload 目录已移入系统
回收站；原因是用户确认改用 W8-base/W4-routed-expert profile。后续只保留新
profile 的 GPTQ 与 AutoRound 验证，避免产物口径混用。

### 4.4 当前 mixed GPTQ no-offload 首层 smoke

历史真实命令让物理 GPU 4～7 可见、前 6 层 staging、仅量化 layer 0、
`nsamples=1`、`seqlen=16`。两个资源策略都是关闭状态：

```text
offload_to_disk           = false
auto_forward_data_parallel = false
```

结果：

| 项目 | 实测值 |
|---|---:|
| 退出状态 | 0 |
| routed expert W4 | 768 |
| attention W8 | 4 |
| shared expert W8 | 3 |
| quantized Linear 合计 | 775 |
| `qweight/qzeros/scales/g_idx` tensor 数 | 各 775 |
| scale dtype | FP16 |
| `qweight/qzeros/g_idx` dtype | INT32 packed / INT32 |
| group size | 64 |
| 对称量化 | true |
| 量化前 staging 大小 | 75.57 GiB |
| 保存后 artifact 大小 | 66.74 GiB（17 shards） |
| 观察到的四卡峰值 | 约 21.9 / 2.7 / 2.6 / 2.6 GiB；单卡容量足够 |

保存后不是只看 `quantize_config.json`：根据每个模块的 `qweight` 与 `scales`
shape 反推物理 pack bits，得到：

```text
768 × routed expert = W4
  4 × attention     = W8
  3 × shared expert = W8
```

provenance 同时记录 `offload_path=null`；artifact 目录没有生成独立 offload
内容。其余三卡只有约 2.6 GiB worker/runtime 占用，不能作为“必须四卡”的证据。
上一次显式开启 forward data parallel 的对照试跑在层尾把完整 MoE layer 复制到
四卡并 OOM，本次关闭后 layer 0 finalize、后续层 replay 与保存均通过。

这仍然只是链路 smoke。`nsamples=1/seqlen=16` 会让部分 down projection 的
damp recovery 偏高，不能拿来评价最终 PPL、KL 或任务精度；正式质量产物必须
使用正常校准规模重新量化。

### 4.5 当前 mixed AutoRound V2 no-offload 首层 smoke

AutoRound 使用同一份前 6 层 BF16 staging 和完全相同的 mixed precision
dynamic rule；只把算法入口切为 AutoRound V2。真实运行参数为：

```text
nsamples                    = 1
seqlen                      = 16
iters                       = 1
max_quant_layers            = 1
offload_to_disk             = false
auto_forward_data_parallel  = false
```

实测结果：

| 项目 | 实测值 |
|---|---:|
| 退出状态 | 0 |
| routed expert W4 | 768 |
| attention W8 | 4 |
| shared expert W8 | 3 |
| `qweight/qzeros/scales/g_idx` tensor 数 | 各 775 |
| layer 0 quantized Linear 合计 | 775 |
| scale dtype | FP16 |
| group size / symmetric | 64 / true |
| artifact | 17 shards，约 67 GiB |
| AutoRound | V2，iters=1 |
| 观察到的四卡显存 | 约 14.4 / 2.4 / 2.4 / 2.4 GiB |

一个 routed expert `gate_proj` 的物理 tensor 为：

```text
qweight [512,2048] INT32-packed  -> W4
scales  [64,2048]  FP16          -> K=4096 / G64
g_idx   [4096]     INT32
```

一个 shared expert `gate_proj` 则是：

```text
qweight [1024,2048] INT32-packed -> W8
scales  [64,2048]   FP16         -> K=4096 / G64
```

因此这不是只依赖 dynamic 正则得出的结论。保存后的 packed shape 也证明了
W4/W8 分界。该任务总耗时约 5 分钟，但 `iters=1` 只适合接口 smoke；正式
AutoRound V2 必须恢复正常迭代数和校准集后再比较 PPL/KL/任务指标。

### 4.6 完整六层 mixed AutoRound V2 no-offload smoke

在 layer-0 验证之后，去掉 `max_quant_layers` 限制，以相同 profile 完成 Block
0～5 的全部量化。任务在物理 GPU 4～7 上退出为 0，保存配置与 provenance
同时记录：

```text
offload_to_disk             = false
offload_path                = null
auto_forward_data_parallel  = false
num_layers                  = 6
method / version / iters    = AutoRound / V2 / 1
```

产物级审计结果：

| 项目 | 实测值 |
|---|---:|
| shard 数 | 6 |
| artifact 文件总量 | 24,199,538,798 bytes（约 22.54 GiB） |
| quantized Linear | 4,650 |
| routed non-shared expert W4 | 4,608 = 6×256×3 |
| attention + shared expert W8 | 42 = 6×7 |
| 每层 W4 / W8 | 768 / 7 |
| `qweight/qzeros/scales/g_idx` | 各 4,650 个 |
| scale dtype | 全部 FP16 |
| group size / symmetric | 64 / true |

W4 与 W8 不是只按模块名推断：routed expert 的 INT32-packed `qweight` 第一维
对应 4-bit packing，shared expert/attention 对应 8-bit packing；shard header 中
所有 4,650 个 scale 均为 FP16。该 smoke 证明完整六层的 capture、replay、
AutoRound、pack 和 save 链路，不代表 `iters=1` 已经达到可交付精度。

## 5. 前 6 个 Block 的结构

| Block | Attention | Compressor | Indexer | Router |
|---:|---|---:|---|---|
| 0 | SWA | 无 | 无 | hash Top-6 |
| 1 | SWA | 无 | 无 | hash Top-6 |
| 2 | CSA | 4 | Top-512 | hash Top-6 |
| 3 | HCA | 128 | 无 | score Top-6 |
| 4 | CSA | 4 | Top-512 | score Top-6 |
| 5 | HCA | 128 | 无 | score Top-6 |

每个 Block 都是：

```text
mHC attn pre
→ RMSNorm
→ SWA / CSA / HCA attention
→ mHC attn post
→ mHC FFN pre
→ RMSNorm
→ routed Top-6 experts + shared expert
→ mHC FFN post
```

## 6. mHC：精确计算与 FP16 结论

输入：

```text
X             [B, P, 4, 4096]
flatten(X)    [B, P, 16384]
fn            [24, 16384]
base          [24]
scale         [3]
mixes         [B, P, 24]
```

先直接调用无 affine 的 RMSNorm，再接 Linear：

```text
normalized = RMSNorm(flatten(X), weight=ones, eps=norm_eps)
mixes      = Linear(normalized, fn, bias=None)
```

这里不用在图中展开 `square -> ReduceMean -> Add -> Rsqrt -> Mul`。XH RMSNorm
接口要求显式 weight，因此用不进入 checkpoint 的全 1 buffer 表达官方的无 affine
RMSNorm；它与原式逐元素等价，HMONNX 中保持为一个 fused `RMSNorm` 节点。attention
前和 FFN 前各执行一次，所以每个 decoder block 由 mHC 新增 2 个 RMSNorm。

拆分：

```text
pre_logits   [B,P,4]
post_logits  [B,P,4]
comb_logits  [B,P,4,4]
```

映射：

```text
pre  = sigmoid(pre_logits * scale[0] + base[0:4]) + hc_eps
post = 2 * sigmoid(post_logits * scale[1] + base[4:8])
comb = Sinkhorn20(comb_logits * scale[2] + reshape(base[8:24],[4,4]))
```

pre collapse：

```text
collapsed[b,p,d] = Σ_j pre[b,p,j] * X[b,p,j,d]
```

post mapping 的方向不能写反：

```text
Y[b,p,k,d] = post[b,p,k] * branch[b,p,d]
           + Σ_j comb[b,p,j,k] * X[b,p,j,d]
```

也就是 `comb.transpose(-1,-2) @ residual`。

### 6.1 少步算法的验证口径

这里把“等价”定义为实际部署边界，而不是只看 4×4 矩阵的平均误差：

1. 同一个官方 checkpoint、相同输入、相同 FP16 runtime，只替换 mHC solver；
2. 先比较真实 mHC logits 上的 4×4 输出和行列边际；
3. 再比较逐层 FP16 hidden 与最终 FP16 logits；
4. 最后用相同 token 计算 NLL/PPL，并检查每个位置的 Top-1；
5. 20 步候选必须先通过自对照 bit-exact，证明 Cache reset、数据和评测链路没有
   引入随机差异。

严格放行条件为：所有被测 FP16 tensor 逐元素一致、Top-1 全同、NLL/PPL 差为 0。
若仅 PPL 接近但 logits 不同，只能另立 approximate/fast mode，不能替换默认实现。

评测代码为：

- `xh2modelzoo/examples_merak/llm/deepseekv4/evaluate_mhc_solvers.py`：first-6
  真实 mHC logits、逐层输出和固定算法搜索；
- `xh2modelzoo/examples_merak/llm/deepseekv4/evaluate_mhc_full_model.py`：官方
  43 层 MP4 checkpoint 的 FP16 logits/PPL 对照；
- 整网 JSON 报告：
  `xh2modelzoo/work_dirs/deepseek-v4-flash-mhc-full43-wikitext4x256-v2.json`。

### 6.2 真实 mHC logits：普通少跑迭代不成立

first-6、一个真实 256-token WikiText-2 prefill 共采集：

```text
12 个 mHC 调用 = 6 层 × (attention + FFN)
3,072 个 4×4 combination matrix
```

相对官方 FP32/20 步：

| solver | comb FP16 元素一致率 | FP16 整矩阵一致率 | comb 最大绝对误差 | 最大边际误差 |
|---|---:|---:|---:|---:|
| FP32 Sinkhorn 10 | 4.20% | 0% | 1.893e-1 | 2.686e-1 |
| FP32 Sinkhorn 16 | 7.66% | 0% | 3.605e-2 | 1.190e-1 |
| FP32 Sinkhorn 18 | 10.36% | 0% | 1.543e-2 | 9.880e-2 |
| FP32 Sinkhorn 20 | 99.994% | 99.902% | 1.788e-7 | 8.474e-2 |
| FP16 Sinkhorn 18 | 4.47% | 0% | 1.554e-2 | 9.882e-2 |
| FP16 Sinkhorn 20 | 22.69% | 0% | 4.867e-3 | 8.490e-2 |

`pre/post` 与 `comb` 要分开理解：前两者不依赖 Sinkhorn 轮数；纯 FP16 计算却会
改变 sigmoid 的舍入，所以“FP16 20 步”本身也不等同于官方 FP32 20 步。上表中
FP32/20 的极小尾差来自显式 stable-softmax 与 PyTorch `softmax` 的 FP32 舍入顺序，
绝大多数在 FP16 边界消失，但不能据此跳过整网验证。

### 6.3 Newton、SOR 和极限解为什么也不能直接替换

评测了两类固定静态候选：

- 3 维 Newton：0/1/2/4/8 次 Sinkhorn warm-up，1～4 次 Newton，是否使用
  column 初始化，4 个 regularization 和 4 个 damping，共 640 组；
- log-domain SOR：2～19 步，`omega=1.00...1.99`，共 1,800 组。

所有候选都保持 stable softmax；搜索过程中没有 NaN/Inf。真实 3,072 个矩阵上的
最佳点为：

| 候选 | 目标 | comb MAE | comb 最大误差 | FP16 整矩阵一致率 |
|---|---|---:|---:|---:|
| 8 warm-up + 2 Newton，reg=0.01，damping=0.5 | 官方 FP32/20 | 7.416e-3 | 7.541e-2 | 0% |
| 12 步 log-SOR，omega=1.24 | 官方 FP32/20 | 2.426e-3 | 2.961e-2 | 0% |

Newton/Birkhoff projection 与 SOR 的优势是更快接近同一个双随机极限，但官方训练
算子是带 `eps` 的**有限 20 步输出**，两者并不是同一目标。用普通 FP32 Sinkhorn
100 步近似极限后，first-6 的 256-token 对照相对 20 步已经出现
`0.2866/token` NLL 差和 91.8% Top-1 一致率。因此“收敛更快”在这里反而会离训练
时算子更远。

[`Accelerating Birkhoff Projection for Manifold-Constrained Hyper-Connections`](https://arxiv.org/abs/2606.07574)
及其[作者实现 `mHC-proj`](https://github.com/yixuan/mHC-proj)中的 Newton 路径还
依赖动态条件数判断、最多 5 次 line search、失败时
回退一轮 log-Sinkhorn 和动态 early-exit；把它无条件裁成固定 2～4 步既不保留
论文算法的稳定性，也不满足当前静态图禁用 `Where/comparison` 的约束。
log-SOR 的参数范围和加速预期参考了
[`A note on overrelaxation in the Sinkhorn algorithm`](https://arxiv.org/abs/2012.12562)，
但本模型的目标是复现有限 20 步，而不是只保证收敛到相同极限。

### 6.4 完整 43 层 FP16 PPL 结果

最终整网验证配置：

```text
checkpoint : 官方 43 层 FP4/FP8 mixed checkpoint，转换为 MP=4
GPU        : 物理 4,5,6,7
runtime    : FP16 activation；同一模型实例只替换 mHC solver
dataset    : WikiText-2 raw test，4 个连续 256-token window
predictions: 1,024 token
cache      : 每个 window 前全部清零，所有候选使用同一输入
```

| mHC solver | FP16 logits 完全一致 | FP16 logit 元素一致率 | Top-1 一致 | NLL/token | PPL | PPL 差 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 Sinkhorn 20（自对照） | 是 | 100% | 1,024/1,024 | 2.116825581 | 8.304732895 | 0 |
| FP16 Sinkhorn 19 | 否 | 0.369% | 965/1,024 | 2.118975550 | 8.322607024 | +0.017874129 |
| log-SOR 12，omega=1.24 | 否 | 0.230% | 945/1,024 | 2.135142595 | 8.458252522 | +0.153519627 |
| FP32 Sinkhorn 100（极限近似） | 否 | 0.134% | 889/1,024 | 2.161400884 | 8.683293432 | +0.378560537 |

16-token 整网烟测也得到同样方向：20 步 PPL 109.403418，19 步为 110.914993，
Top-1 仅 15/16 相同。4×256 是用于当前决策的主报告；它仍不是发布级完整
WikiText-2 benchmark，但已经足以否定“两个 FP16 推理完全一样”这一零差异命题。

当前结论：

- logits 先减 row max，FP16 20 步在采样范围内没有 NaN/Inf；
- 普通 19 步都无法通过，更不能把迭代直接砍到 4～8 次；
- Newton/SOR/极限解不等于训练时有限 20 步，也无法通过 FP16 logits/PPL 门禁；
- **默认实现继续保留 FP16 stable Sinkhorn 20**；不加入未经验证的 fast mode；
- 在不能做专用融合 kernel 的前提下，目前没有已验证的“步数更少且等价”替代。

### 6.5 用一个大算子封装精确 20 步，而不是改算法

少步 solver 不能通过零差异门禁，但 20 步不应在 ONNX 中展开成上千个通用节点。
因此 xhquant 新增 `XH2a::MHCSinkhorn`，属性固定携带
`sinkhorn_iters=20`、`eps=1e-6`，内部仍严格执行官方顺序：

```text
stable row softmax + eps
-> first column normalization
-> 19 x (row normalization -> column normalization)
```

算子接受 `[...,N,N]` 的通用方阵，V4 实际使用 `[B,P,4,4]`。eager XH、XH2a
module、HMONNX parser 已对随机输入和极端 logits 与独立官方展开实现做逐 bit 对照；
默认路径没有换 Newton/SOR，也没有减少迭代次数。这样同时满足两件事：数值仍是
训练时的有限 20 步，导出器只看见一个不可展开节点。

真实 first6 的 prefill/decode 图各有 6 层 attention mHC 和 6 层 FFN mHC，即各
12 个 `MHCSinkhorn`。融合前后节点数为：

| 图 | 展开 Sinkhorn20 | 单节点 Sinkhorn20 | 减少 |
|---|---:|---:|---:|
| prefill-256 | 5,940 | 4,476 | 1,464 |
| decode-1 | 5,928 | 4,464 | 1,464 |

first6 主图 ONNXScript optimization 实测约 11/13 秒；此前 full43 展开图在
`_collapse_slices` 中运行四小时仍不能完成。大算子解决的是图规模和导出迭代速度，
不是用近似算法掩盖数值差异。后端若将来提供融合 kernel，只需实现同一个算子 ABI，
不需要重新改模型图。

## 7. SWA：固定 256 prefill 的 Cache

每个 token 的主 KV latent 为 512：

```text
NoPE 448 + partial-RoPE 64
```

数学上同一个 `kv` 同时作为 K 和 V。XH2 上：

```text
SWA K Cache  [B, 1, 384, 512]
SWA V Cache  [B, 1, 384, 512]
```

这里必须区分两个相近但不等价的长度：

```text
persistent backing = align(sliding_window + prefill_chunk_length, 16)
                   = align(128 + 256, 16) = 384

attention view     = align(sliding_window + input_length - 1, 16)
prefill            = align(128 + 256 - 1, 16) = 384
decode             = align(128 +   1 - 1, 16) = 128
```

不能只开 128，因为一次 graph call 需要同时容纳旧窗口和 256 个当前 token。也不需要开 512；原先“prefill chunk + prefill chunk”的 512 只是保守绕过，准确的持久
ABI 是 384。prefill/decode 两张图共享这份 384 backing，但 `KVcache` 输出给
Attention 的视图分别是 384 和 128。

不能因为默认配置下两个 prefill 公式碰巧都对齐为 384，就在代码里复用同一个
公式。边界例 `window=8,input=9,alignment=16` 会明确暴露差异：

```text
backing        = align(8 + 9, 16)     = 32
attention view = align(8 + 9 - 1, 16) = 16
```

因此 `StaticSWAUpdate`、静态 Cache ABI 和 Host mask 分别调用 backing/view
专用 helper。最终 HMONNX 审计还会逐层核对 K/V 图输入等于持久 backing、
`swa_attention_mask` 与 `KVcache` 输出等于当前 stage 的 attention view，并验证
`KVcache(axis=2, attention_max_length=window)`；任一层只开 16 backing 都会在导出时
硬失败。

不在图内根据 `past_length/current_length` 重新计算可见位置，也不做逐 query
的动态 window gather。Host 已经知道本次调用的绝对位置和有效 token 数，直接生成
FP16 additive mask：

```text
prefill: swa_attention_mask [B,1,256,384]
decode : swa_attention_mask [B,1,  1,128]

valid   = 0
invalid = -65504
```

图内只做 `attention_score + swa_attention_mask`。这样位置逻辑不会导出成
`Less / And / Where / Clip`；384/128 行物理 K/V 直接送 attention。padding query
对应的整行 SWA mask 都是无效值，attention sink 仍保持有效，因而 softmax 不会出现
全 `-inf` 行。

## 8. CSA：固定形状重叠压缩

### 8.1 投影和 state

主 compressor：

```text
kv_proj(x)      [B,P,1024] = [Ca(512), Cb(512)]
gate_proj(x)    [B,P,1024]
position_bias   [4,1024]
kv_state        [B,8,1024]
score_state     [B,8,1024]
offset          [1], Host 当前调用输入，取值 1..4
```

Indexer compressor 同构，只把 512 换成 128：

```text
kv/state width  = 256
pooled K width  = 128
```

### 8.2 prefill=256 的 `[B,64,8,512]`

初始 state 的 KV 为 0，score 为 `-65504`。第一次调用 `past_length=0`，所以 Host
计算出的 `offset=4`。

```text
cat_kv    = concat([kv_state, current_kv], dim=1)       # [B,264,1024]
cat_score = concat([score_state, current_score], dim=1) # [B,264,1024]

aligned = DynamicSlice(cat, start=offset, length=260)
aligned = reshape(aligned, [B,65,4,1024])

left  = aligned[:,0:64,:,0:512]     # [B,64,4,512]
right = aligned[:,1:65,:,512:1024]  # [B,64,4,512]

candidate = concat([left,right], dim=2) # [B,64,8,512]
```

Gate 完全同样切片。softmax 维度是候选轴 8，而不是 feature 轴：

```text
weight[b,g,r,d] = softmax(score[b,g,:,d], dim=候选8)[r]
pooled[b,g,d]   = Σ_r weight[b,g,r,d] * candidate[b,g,r,d]
```

因此每个 feature `d` 都可以选择不同的历史 token。

### 8.3 为什么首轮 `-65504` 不会出错

第一轮 state 的 8 行 score 都是 `-65504`，但 offset=4，动态 slice 从 state 后半开始：

- 第一个输出的左 4 个候选无效；
- 右 4 个候选正好是 token 0～3；
- softmax 给左侧的权重数值上接近 0；
- 即便某个固定输出槽全部无效，该槽也不会写 Cache。

有效输出数由 Host 根据绝对长度计算：

```text
pending   = past_length % 4
new_count = (pending + current_length) // 4
offset    = 4 - pending
validity[g] = 1 if g < new_count else 0, g=0..ceil(P/4)-1
```

这三个值分别作为全体 CSA layer 共享的
`csa_compressor_new_count/csa_compressor_offset/csa_compressor_validity` 输入。
`pooled` 先乘 FP16 0/1 validity，`LLMCache` 再只提交前 `new_count` 行。图内不再通过
`Less/LessOrEqual/Where` 判断固定输出槽是否有效。不能把“全无效 softmax 会输出
什么”当作 Cache 有效性的判定。

### 8.4 下一个 state

不需要按 token 循环更新 8 行 state：

```text
next_state = DynamicSlice(
    concat([old_state, current_projection], dim=1),
    start=current_length,
    length=8,
)
```

它总是取“截至当前有效 token 的最后 8 行”，不足 8 行时仍保留左侧初始无效行。
只有 KV/score carry state 跨调用保存；下一次的 offset/new-count/validity 全部由
Host 用新的绝对 `past_length` 重新生成，不是 graph output。

压缩 Cache 写入：

```text
write_start = past_token_length // 4
write_count = new_count
```

经过 RMSNorm 后，最后 64 维做 compressed RoPE，位置为：

```text
4 * (write_start + arange(new_count))
```

### 8.5 CSA Indexer

Indexer query：

```text
q_residual        [B,P,1024]
index q_b_proj    [B,P,64,128]
partial RoPE      last 64
```

Indexer key Cache：

```text
K_index [B,65536,128]
```

它是 K-only Cache，不需要 V。

对 query `t` 和压缩条目 `s`：

```text
dot[b,t,h,s] = q_index[b,t,h,:] · K_index[b,s,:]

score[b,t,s] = Σ_h (
    ReLU(dot[b,t,h,s])
    * weights_proj(x)[b,t,h]
) / sqrt(128) / sqrt(64)
```

因果有效条件由 Host 计算：

```text
s < floor((absolute_query_position + 1) / 4)
```

Host 传入：

```text
csa_index_validity [B,P,65536], FP16, valid=1, invalid=0
```

图内不改变有效 score，先由同一批 query/key/weight 计算一个保守下界。因为：

```text
relu(q·k) <= max(abs(k)) * sum(abs(q))

bound = scale * max(abs(K_cache))
              * sum_h(abs(index_weight_h) * sum_d(abs(q_h,d)))

invalid_floor = -(4 * bound + 1)
score = score * validity + invalid_floor * (1 - validity)
```

`-bound` 已经是有效 score 的数学下界，额外 4 倍间隔覆盖 FP16 reduction 的舍入。
因此无效槽严格低于有效槽，不依赖 checkpoint 的经验值域，也不需要
`Less / Where / Clip`。更重要的是，有效 score 完全不变，不会因为有界单调变换在
FP16 下饱和并制造额外 tie。

TopK 默认降序排序。若当前有效压缩条目数为 `N < 512`，前 `N` 个结果有效、后
`512-N` 个是哨兵补位；因此 Host 可以直接按 `min(N,512)` 构造 TopK 结果区域的
additive mask，不需要图内再 gather validity。这里仍然必须有 TopK 前 validity，
否则尚未写入的零 Cache 可能与有效零分发生 tie 并挤占有效条目。

### 8.6 为什么要 tiled TopK

最大上下文下，一次性 score tensor 为：

```text
[B,256,64,65536]
```

仅 BF16 就约 2 GiB/层，而且还没有计算 main attention。当前静态实现按 1024 个压缩条目一块：

1. 计算 `[B,P,64,1024]`；
2. ReLU、乘 head weight，并立刻 reduce head；
3. 取 tile 内 Top-512；
4. 与 running Top-512 拼成 1024；
5. 再取 Top-512；
6. 静态重复 64 个 tile。

该算法与一次性全局 TopK 等价，只有精确相同分数的 tie 顺序未定义。当前已有单测与 full TopK 对齐。

### 8.7 main attention

TopK 后分别 gather 物理 K/V：

```text
CSA K selected [B,P,512,512]
CSA V selected [B,P,512,512]
SWA K physical [B,Lswa,512]
SWA V physical [B,Lswa,512]
```

score/value 逻辑拼接，其中 `Lswa=384`（prefill）或 `128`（decode）：

```text
prefill key width = 512 + 384 = 896
decode  key width = 512 + 128 = 640

csa_attention_mask:
prefill [B,1,256,896]
decode  [B,1,  1,640]
```

每个 head 的 logits 还要加一个 attention sink。sink 只有 logit，没有对应 V，由
`SinksSoftmax` 单独接收，不包含在 Host mask 中；所以含 sink 的 prefill/decode
逻辑宽度分别为 897/641。

## 9. HCA：ratio 128

HCA 没有 overlap，也没有 Indexer：

```text
kv_proj(x)      [B,P,512]
gate_proj(x)    [B,P,512]
position_bias   [128,512]
kv_state        [B,128,512]
score_state     [B,128,512]
offset          Host 当前调用输入，取值 1..128
```

固定 P=256 时最多产生 2 个条目：

```text
pending   = past_length % 128
new_count = (pending + current_length) // 128
offset    = 128 - pending
validity[g] = 1 if g < new_count else 0, g=0..ceil(P/128)-1
```

三者由 Host 作为全体 HCA layer 共享的
`hca_compressor_new_count/hca_compressor_offset/hca_compressor_validity` 输入。第一次
调用 `past_length=0`，因此 offset 为 128；图只回传 128 行 KV/score carry state。

动态 slice 后 reshape：

```text
[B,2,128,512]
```

softmax 仍在 token/candidate 轴 128 上逐 feature 计算。Cache write start 为 `past_token_length // 128`。压缩 RoPE 位置为：

```text
128 * (write_start + arange(new_count))
```

HCA attention 使用所有因果有效的压缩条目，不做 TopK。最大 2,048 个条目，可以直接对固定 Cache 做 dense/broadcast matmul；不要扩成 `[B,P,2048,512]` 的 K/V 实体副本。

有效压缩前缀同样由 Host 按
`floor((past_length + query_local + 1) / 128)` 生成，并与 SWA mask 拼接：

```text
hca_attention_mask:
prefill [B,1,256,2048+384] = [B,1,256,2432]
decode  [B,1,  1,2048+128] = [B,1,  1,2176]
```

所以 HCA 图也不再包含用于重建因果有效区的 `Less / And / Where`。

## 10. 256 Ki 固定 Cache ABI

`max_context_length = 262144` 时，每个相应 layer/request：

| 张量 | shape | 说明 |
|---|---|---|
| SWA K | `[B,1,384,512]` | 物理输入 backing |
| SWA V | `[B,1,384,512]` | 与 K 分离 |
| CSA main K | `[B,1,65536,512]` | 262144/4 |
| CSA main V | `[B,1,65536,512]` | 与 K 分离 |
| CSA Index K | `[B,1,65536,128]` | K-only |
| CSA main KV state | `[B,8,1024]` | FP32/FP16 待硬件定 |
| CSA main score state | `[B,8,1024]` | 初始 -65504 |
| CSA Index KV state | `[B,8,256]` | Indexer |
| CSA Index score state | `[B,8,256]` | Indexer |
| HCA K | `[B,1,2048,512]` | 262144/128 |
| HCA V | `[B,1,2048,512]` | 与 K 分离 |
| HCA KV state | `[B,128,512]` | partial group |
| HCA score state | `[B,128,512]` | partial group |

如果 main K/V 都按 1 byte/element 粗略估算，不含 scale/alignment：

- 单个 CSA layer main K+V：64 MiB；
- 单个 CSA layer Index K：8 MiB；
- 单个 HCA layer K+V：2 MiB；
- 单层 SWA 384 backing K+V：384 KiB。

最终显存必须按真实 K-SEFP/V-SEFP scale layout、对齐和 layer 数重新核算，不能继续沿用“逻辑 KV 共享所以只存一份”的估算。

按当前 ABI 精确计算，scale 使用 FP16、value 使用 MXINT8、block64，且不计
allocator/compiler 的额外对齐：

| 网络 | CSA/HCA 层数 | FP16 K/V/Index Cache | MXINT8 + FP16 scale | compressor state FP16 |
|---|---:|---:|---:|---:|
| first6 | 2 / 2 | 0.2935 GiB | 0.1513 GiB | 0.0006 GiB |
| full43 | 21 / 20 | 3.0627 GiB | 1.5792 GiB | 0.0057 GiB |

这是 batch=1、单 request、完整 256 Ki context 的 backing allocation。189 个
full43 物理 Cache tensor 已包含每层 SWA K/V、CSA main K/V + Index K、HCA
main K/V，不能再乘一次“KV 两份”；表中本来就已经物理分开。

## 11. K/V 数学共享与物理分离

模型的主 attention 确实是：

```text
K_latent == V_latent
```

官方 eager/vLLM 可以让同一个 512-d row 同时参与 QK 和 probability×V。但 XH2 的目标编码是：

- K-SEFP：scale/exponent 沿 head-dim 分组；
- V-SEFP：scale/exponent 沿 token 轴分组。

因此写 Cache 时应从同一个 BF16 latent 分叉：

```text
latent ──→ K quant/layout ──→ physical K Cache
       └─→ V quant/layout ──→ physical V Cache
```

SWA、CSA main、HCA main 都这样处理。CSA Indexer 只做 query-key score，所以仅存 K。

若 target Cache 使用 block64 MXINT8/SEFP，scale 也不能共用：

```text
K scale: [B,H,T,ceil(D/64)]  # 沿 head-dim 归约
V scale: [B,H,ceil(T/64),D]  # 沿 token 归约
```

当前 Merak bring-up 图先用 FP16 Cache 验证语义和状态更新，已经把 K/V 物理
tensor 分开；MXINT8 编码、scale tensor 和 Cache kernel lowering 是下一阶段，
不能因为 `K_latent == V_latent` 而在 ABI 中重新合并。

## 12. MLA 权重吸收/高效形态

V4 checkpoint 已经是高效 latent 形态：

```text
q_a_proj → q_a_norm → q_b_proj
shared kv_proj → kv_norm
grouped o_a_proj → o_b_proj
```

适配时禁止：

- 展开成 64 份独立 K/V projection；
- 为每个 head 存 512-d K/V；
- 把 grouped `o_a_proj` 当普通二维 Linear 误替换；
- 为方便量化把 `o_a/o_b` 离线乘成巨大的 dense output projection。

可做的“吸收”主要是 kernel 级融合：

- `q_a + norm + q_b + q_norm + RoPE`；
- `kv_proj + norm + RoPE + K/V 两路 Cache quant/write`；
- inverse RoPE + grouped `o_a` BMM + `o_b`；
- compressor `wkv/wgate` 并行 GEMM；
- Indexer q RoPE/Hadamard/cache quant 与 scorer 融合。

这保持代数结构，不增加 KV Cache 维度。

### 12.1 RoPE：离线表 + Slice + XH Rope

RoPE 不应在运行图中根据 `valid_length` 重算 `Cos/Sin`，也不需要先做
`Cast -> Reshape -> Add` 来拼位置。模型 wrap 时已经知道最大上下文 `Cmax` 和固定
chunk `P`，因此一次性生成四组逻辑位置表：

```text
main raw       position = 0 .. Cmax+P-2
compress raw   position = 0 .. Cmax+P-2
CSA sampled    position = 4   * arange(Cmax/4 + ceil(P/4))
HCA sampled    position = 128 * arange(Cmax/128 + ceil(P/128))
```

运行时只做连续动态切片：

```text
main/compress = DynamicSlice(raw_table, past_length, P)
CSA           = DynamicSlice(csa_sampled_table, csa_write_start, ceil(P/4))
HCA           = DynamicSlice(hca_sampled_table, hca_write_start, ceil(P/128))
```

表的离线生成还要满足字节确定性。本机 PyTorch 2.8 的首个大规模 CPU `torch.cos`
调用存在一次性 lazy-dispatch 舍入漂移：同一输入约一万个 FP16 元素会偶发相差
1 ULP，导致普通/低内存导出的 external data 无法 `cmp`。当前在每次离线表生成前
用固定 4096 元素做一次丢弃式 `cos/sin` 预热，再计算真实表；4 个全新进程得到相同
SHA256。预热发生在 wrap 阶段，不进入 forward/HMONNX，也不产生运行时节点。

`past_length/current_length/*_write_start/*_offset/*_new_count` 的 Host ABI
统一为 INT32 `[1]`。XH `KVcache`、`DynamicSlice`、`Gather` 都原生接受 INT32，
因此这些输入直接接算子，不再生成把 `[1]` reshape 成 `[1]`、再 cast 回 INT32 的
空操作。前三层 hash router 的 INT32 `input_ids` 也直接用于 `tid2eid` Gather。

最终图同时禁止 `ConstantOfShape`。SWA/CSA/HCA 的补齐区、carry state 和最大 Cache
容量都在 wrap 阶段按 `Cmax`、固定 prefill chunk 和 decode 长度注册为静态 buffer；
运行时只允许用已有张量的标量零值做广播，不允许 `new_zeros/new_full` 一类按运行时
shape 创建张量。HMONNX 终检将 `Cast` 和 `ConstantOfShape` 都列为硬失败项，prefill
与 decode 任一图出现一个都不能交付；扫描范围覆盖顶层图、节点属性里的嵌套
`GraphProto` 以及 ONNX local function，不能把动态分配藏到子图中规避门禁。

CSA/HCA 的绝对位置虽然在原始 token 轴上不连续，但预采样后在各自 Cache 轴上连续，
所以 `Slice` 比运行时 `Gather` 更简单；若后续出现无法预采样成连续表的位置族，再用
Host 提供 index 的 `Gather`，而不是在图内生成 index。表尾额外保留 `P-1` 或对应
压缩输出数，使最后一个不足 P、经 padding 的 chunk 也不会越界；无效行仍由 Host
mask 禁止参与计算和写 Cache。

DeepSeek-V4 的 64 维 partial RoPE 使用交错复数布局
`[real0,imag0,real1,imag1,...]`，而 XH Rope 接口使用 half-split 布局
`[all_real,all_imag]`。因此在一个 `XH2a::Rope` 前后各做固定 reshape/transpose
置换；正向 Q/K/compressor 和输出侧 inverse RoPE 都复用这一 wrapper。最终图应当
没有任何域的 `Cos/Sin`，首六层固定算子数为 26，完整 43 层为 212。

### 12.2 MoE 路由 ABI：全专家分数 + 已选专家 ID

`MoeBlock` 的稳定接口为：

```text
hidden_states       [B,P,4096]
all_expert_scores   [B,P,256] FP16
selected_expert_ids [B,P,6]   INT32（hash）/ INT16（XH TopK）
```

第二输入不能预先 gather 成 `[B,P,6]`。V4 先计算
`scores=sqrt(softplus(router_logits))`；score-router 用
`scores+correction_bias` 选 ID，hash-router 用 `tid2eid[input_ids]` 选 ID，但两者
真正参与加权的都仍是未加 bias 的 `scores`。因此 router 返回完整原始分数和单独的
ID，`MoeBlock(topk_outside=True, normalize_routing_weights=True)` 内部按 ID gather、
Top-6 归一化，层外再乘 `routed_scaling_factor=1.5`。最终 HMONNX 审计会拒绝第二、
第三输入都为 K 维的旧接口，并要求每层的主计算/`+1` correction 两个 MoeBlock
都满足这一契约。

前三层 hash 路由直接 Gather checkpoint 的 `tid2eid`，所以 ID 为 INT32；后续
score 路由的 XH TopK 原生输出 INT16（256 个 expert 足够表示）。两者都是
`MoeBlock` 已支持的整数 ID ABI，不应为了表面统一插入 Cast。

## 13. 三个输入长度示例

### 13.1 78 token

只调用一次 prefill256 graph：

```text
current_length = 78
padding        = 178
CSA new        = 19
CSA residual   = 2 token，下一调用 Host offset=2
HCA new        = 0
HCA residual   = 78 token，下一调用 Host offset=50
```

padding token 不写 SWA/CSA/HCA Cache，padding query 输出也不参与下一层有效 token。

### 13.2 2,359 token

```text
9 × 256 + 55
总 graph calls = 10
CSA entries    = floor(2359/4)   = 589
HCA entries    = floor(2359/128) = 18
CSA residual   = 3，下一调用 Host offset=1
HCA residual   = 55，下一调用 Host offset=73
```

最后一次仍送 `[B,256,...]`，但 `current_length=55`。CSA 最后只写 13 个条目，HCA 最后不写新条目。

### 13.3 43,456 token

```text
169 × 256 + 192
总 graph calls = 170
最后 chunk start = 43,264
CSA entries       = 10,864
HCA entries       = 339
CSA residual      = 0，下一调用 Host offset=4
HCA residual      = 64，下一调用 Host offset=64
```

最后 chunk 写 48 个 CSA 条目和 1 个 HCA 条目。

上述 offset 只是用总有效长度为下一次 graph call 现算出的对齐量，并不从当前图的
输出回灌。真正回灌的只有 compressor KV/score carry state。

## 14. 已落地的 XH2 静态组件

核心代码位置：

```text
xhmodel_merak/xh_llm/models/deepseek_v4/mhc.py
xhmodel_merak/xh_llm/models/deepseek_v4/compressor.py
xhmodel_merak/xh_llm/models/deepseek_v4/indexer.py
xhmodel_merak/xh_llm/models/deepseek_v4/host_masks.py
xhmodel_merak/xh_llm/models/deepseek_v4/static_attention.py
xhmodel_merak/xh_llm/models/deepseek_v4/static_cache.py
xhmodel_merak/xh_llm/models/deepseek_v4/full_model.py
xhmodel_merak/xh_llm/models/deepseek_v4/runtime.py
xhmodel_merak/xh_llm/models/deepseek_v4/deepseek_v4_model.py
xhmodel_merak/xh_llm/models/deepseek_v4/deepseek_v4_hmonnx_inference.py
examples_merak/llm/deepseekv4/export_hmonnx.py
```

当前包含：

- mHC pre/post 与稳定 Sinkhorn20；
- CSA ratio4 overlap 固定形状 transition；
- HCA ratio128 固定形状 transition；
- padded output 不写固定 Cache；
- 256 Ki Cache shape contract；
- 分离 K/V gather；
- CSA tiled global TopK；
- 78/256/2359/43456/262144 host chunk plan；
- MLA shared latent 与 grouped output projection，不展开 KV heads；
- hash/score Router 与精确 clamped-SwiGLU；
- GPTQModel/AutoRound `quant_weight` 传递：routed MoeBlock W4，其余 Linear W8；
- prefill/decode 两张图共享 packed expert Parameter/storage，不复制巨型专家权重；
- HMONNX runtime 原位提交 compressor state，保持静态输入 buffer 地址稳定；
- SWA/CSA/HCA 的位置有效区全部由 Host 一次构造并跨同类层共享；attention 图接收
  FP16 additive mask，CSA Indexer 接收 FP16 0/1 validity；
- CSA/HCA 的 compressor `validity/new_count/offset` 也由 Host 根据绝对
  `past_length/current_length` 构造，同类层共享，不再在图内比较 token 位置；
- meta 导出时只让 256 Ki 持久 K/V Cache 使用 meta 形状占位；CSA/HCA 的
  compressor KV/score carry state 仍是普通 CPU 图输入，避免 meta/CPU 运算混用；
- HCA 无完整 128-token group 的固定 compressor 输出仍不会写入 Cache；padding
  query 的 attention token 全由 Host mask 屏蔽，sink 保证 softmax 数值稳定；
- no-offload 默认配置和 CLI 显式 opt-in。

首六层图的固定 ABI：

```text
inputs       = 50 = 6 个模型/调度输入 + 10 个 Host 位置/压缩控制 + 34 个 Cache/state
outputs      = 13 = logits(1) + CSA carry states(2×4) + HCA carry states(2×2)
LLMCache ops = 22 = SWA(2×2) + CSA(2×5) + HCA(2×4)
```

10 个跨层共享的 Host 输入为：

```text
swa_attention_mask
csa_index_validity
csa_attention_mask
hca_attention_mask
csa_compressor_validity
csa_compressor_new_count
csa_compressor_offset
hca_compressor_validity
hca_compressor_new_count
hca_compressor_offset
```

其中 offset 表示当前调用在旧 carry state 中的动态切片起点，不是持久 Cache；CSA
main 与 Indexer 共用同一个 ratio-4 offset。输出只保留每个 CSA layer 的 main/index
KV、score 四个 state，以及每个 HCA layer 的 KV、score 两个 state。

tiny 首六层已完成：

```text
TorchFX frontend
→ XH2a quant graph (w8a16h1_sefp)
→ aligned PTQ
→ fixed graph
→ export graph
→ HMONNX translation/post transforms
```

初版“图内生成位置 mask”的验证图为 46 输入、23 输出。第一次外提四个 attention
mask/validity 后是 50/23，但仍把 offset 当作每层 state，并在图内比较 compressor
输出槽。最终版本再增加 6 个跨层共享 compressor control，同时删除 6 个逐层 offset
输入，所以输入数仍是 50；删除所有 offset/new-count 输出后，输出数由 23 降为 13。
这些历史图均不再作为交付基线。

当前真实六层 AutoRound packed checkpoint 的最终普通导出结果如下：

| 项目 | prefill | decode |
|---|---:|---:|
| 输入 shape | `[1,256,4096]` | `[1,1,4096]` |
| 图输入 / 输出 | 50 / 13 | 50 / 13 |
| 四个 attention Host 输入 | SWA `[1,1,256,384]`；CSA validity `[1,256,65536]`；CSA attention `[1,1,256,896]`；HCA `[1,1,256,2432]` | SWA `[1,1,1,128]`；CSA validity `[1,1,65536]`；CSA attention `[1,1,1,640]`；HCA `[1,1,1,2176]` |
| compressor validity | CSA `[64]`；HCA `[2]` | CSA `[1]`；HCA `[1]` |
| compressor count/offset | CSA/HCA 各 `[1]` int32 | CSA/HCA 各 `[1]` int32 |
| 节点数 | 5,940 | 5,928 |
| initializer | 656 | 646 |
| `KVcache` | 22 | 22 |
| W4 `MoeBlock` | 12 | 12 |
| W8 `Linear` | 126 | 126 |
| fused XH `RMSNorm` | 50 | 50 |
| XH `Rope` | 26 | 26 |
| FP16 `Clip` | 12 | 12 |
| `Cast/ConstantOfShape` | 0 / 0 | 0 / 0 |
| `Shape/Size/Range/NonZero` | 0 | 0 |
| `Where/Less/LessOrEqual/Greater/GreaterOrEqual` | 0 | 0 |
| HMONNX 主文件 | 18,357,522 bytes | 18,349,194 bytes |
| external data | 41,381,618,688 bytes（38.539636 GiB） | 41,381,544,192 bytes（38.539566 GiB） |
| SHA256（主文件） | `47d771b6548ad0217061fddceb80c38533857799473525a128712d0ede167437` | `21372eebeb52b846c3551e860474f31c7203d8821e553b025ac717a612f28aa6` |
| MD5（主文件） | `b213c39a3073b90fc0ab2e1c1a01ed87` | `d102b72195bf7a107cb3c7af4b8681cb` |

本轮仅保留下面四套 AutoRound/GPTQ 最终产物；v16～v25 及有 RoPE 漂移的 GPTQ
v28 中间导出已删除：

```text
xh2modelzoo/work_dirs/deepseek-v4-flash-first6-autoround-regular-final-v27
xh2modelzoo/work_dirs/deepseek-v4-flash-first6-autoround-lowmem-final-v26
xh2modelzoo/work_dirs/deepseek-v4-flash-first6-gptq-regular-final-v29
xh2modelzoo/work_dirs/deepseek-v4-flash-first6-gptq-lowmem-final-v30
```

源 checkpoint 是第 4.6 节审计过的完整六层 AutoRound 产物，不再是“仅 layer 0
量化”的旧 smoke。当前普通/低内存导出分别约耗时 1,356/1,331 秒；两条路径都只
暴露一张物理 GPU（本轮分别为 6/5）、`enable_auto_offload=false`、
`packed_checkpoint=true`，GPU 峰值均约 2.63 GB。

### 14.1 为什么整图仍有 `Clip=12`

这 12 个 `Clip` 不是 mask、position 或 compressor validity。官方配置的
`swiglu_limit=10`，每层都有一个始终执行的 shared expert，其激活严格为：

```text
gate = Clip(gate_proj(x), min=-inf, max=10)
up   = Clip(up_proj(x),   min=-10, max=10)
y    = down_proj(SiLU(gate) * up)
```

因此 6 个 block × 每层 gate/up 两路 = 12 个独立 FP16 `Clip`。它们不能直接删除，
否则就改变官方模型语义。routed experts 也使用相同的 limit，但已融合进自定义
`MoeBlock(limit=10)`，不会表现为独立 ONNX `Clip`。当前审计只允许 FP16 且边界
精确为 `[-inf,10]` 或 `[-10,10]` 的这两类 Clip；其他 dtype、其他边界或任意
`Where/Less/LessOrEqual/Greater/GreaterOrEqual` 都直接判错。若后端需要减少节点，
正确方向是增加 fused `ClampedSwiGLU`，而不是去掉 clamp。

### 14.2 普通/低内存导出字节门禁

相同 checkpoint/config 下，已对下面四项执行 `cmp -s`，结果全部一致：

```text
prefill ONNX main          identical
prefill external data      identical
decode ONNX main           identical
decode external data       identical
```

这组门禁分别在 AutoRound `v26(lowmem)/v27(regular)` 和 GPTQ
`v30(lowmem)/v29(regular)` 上通过。GPTQ 初次低内存 `v28` 正是因为命中了前述
首个 `torch.cos` 的 1-ULP 漂移而被判失败并删除，证明这里检查的是 external data
实字节，而不是只看主图 hash。

这覆盖了图结构、initializer 顺序、external-data offset/length、量化 weight、FP16
scale 和其他参数的最终序列化结果，不只是“相同输入下输出接近”。低内存路径主进程
峰值约 36.6 GiB，普通路径约 158.0 GiB；低内存导出不改变最终 HMONNX 字节。

同一产物也完成了 xhquant 低内存 loader/解析验证：W4 runtime contract 识别两图
各 12 个 `MoeBlock`、18 个实际 W4 initializer，将 36 个图引用归并为 18 个共享
packed tensor；单图 W4 source 36 GiB、runtime packed 18 GiB。prefill/decode 共
601 对 shared initializer，运行时唯一 initializer 约 20.4684 GiB。SHA256 cache
命中后 loader 为 18.804 秒，prefill/decode parser 分别为 1.206/1.209 秒。为此
loader/parser 的统一约束是：initializer 可以是
NumPy tensor，也可以是 mmap-backed `SharedTensor`；parser 不得私自假定
`torch.from_numpy(...)`，否则会破坏低内存共享语义。

packed-only 路径新增的关键约束是：

- 普通 Linear 直接把 GPTQModel packed code 解包成目标整数布局，并沿用 FP16
  G64 scale；不把反量化后的 BF16 weight 当量化输入再走一次 PTQ；
- routed expert 的 gate/up/down 每层各只打包一次；主分支、correction 分支、
  prefill 图和 decode 图通过 Parameter/storage 共享引用；
- clamped-SwiGLU 的 correction 仍是独立 `MoeBlock` 计算，但 up projection 为
  constant-one，不再为 correction 重复导出一份巨型 up weight；
- 图/device guard 遇到 meta Cache placeholder 时不做 CPU/GPU 实体搬运；
  prequant weight 已对齐后也不再做无意义的 device round-trip；
- prefill HMONNX 文件落盘后立即释放对应 ExportedProgram 并回收 allocator，
  再开始 decode trace，避免 43 层时同时持有两份 initializer-sized 导出对象。

与旧 dense-bridge 首六层导出的对比：

| 指标 | 旧 dense bridge | packed-only | 变化 |
|---|---:|---:|---:|
| 每张图 external data | 63.2207 GiB | 38.4707 GiB | -39.15% |
| 主进程 CPU 峰值 | 319.512 GiB | 158.525 GiB | -50.39% |
| 大尺寸 MoE qweight initializer | 30 | 18 | -40% |
| GPU process 峰值 | 约 22.77 GiB | 约 4.26 GiB | -81.3% |

### 14.3 完整 43 层的逐 MoE 低内存导出

普通 packed-only 路径虽然不再保留 BF16 dense bridge，但仍会同时构造所有层的
MoE packed tensor；43 层时 CPU 峰值仍不理想。现在默认采用与 Qwen3.5
`BigHFModelExportHelper` 同一类的主图/子图拆分，但 V4 的边界只选完整 MoE：

```text
main prefill/decode graph
  attention + mHC + CSA/HCA + cache + norm/head
  blocks.layers.N.mlp = PlaceHolder(hidden_states, input_ids) -> hidden_states

for N = 0..42:
  model.layers.N.mlp packed tensors from safetensors
  -> GPTQModel v1/v2 normalization
  -> signed int4/int8 + FP16 G64 scale（不生成 BF16 expert weight）
  -> DeepSeekV4MoE
  -> export prefill-256 subgraph
  -> export decode-1 subgraph
  -> release to meta + gc

main HMONNX + 43 MoE subgraphs -> inline -> final HMONNX
```

选择 MoE 而不是整个 decoder layer 的原因是：MoE 占绝大多数权重，同时接口只有
`hidden_states [B,P,4096]`、`input_ids [B,P]` 两个输入和一个输出；CSA/HCA 的
Cache 更新、compressor overlap state、RoPE 与多输出 ABI 全留在主图，不需要跨
子图重接几十个端口。Hash-MoE 的 `input_ids` 也显式保留，前三层不会退化成普通
TopK router。

当前真实探针（完整六层 AutoRound packed artifact）为：

| 阶段 | 结果 |
|---|---:|
| HF/Defuser meta packed shell | 4,650 个，全部 qweight 在 meta，RSS 2.32 GiB |
| 主图 materialize | 6 个零参数 MoE placeholder，24 个 attention packed W8 |
| 主图 static conversion 峰值 | RSS 3.60 GiB |
| 单个 layer-0 MoE load audit | 1 parameter + 3,085 buffers，771 packed Linear |
| Hash router state | `tid2eid [129280,6]` + learned router weight |
| routed expert | 768 个 W4G64，scale FP16 |
| shared expert | 3 个 W8G64，scale FP16 |
| 单 MoE load/convert/pack | 5.31 s，峰值 RSS 19.43 GiB |
| release | 退回 meta + gc，0.88 s |

这条路径不是 offload：没有启用 `device_map=auto`、Host/NPU 换入换出或磁盘
offload。它只是导出期的确定性图分解，任一时刻只物化一个 MoE。与仓库现有
Qwen3.5 大模型导出保持同一入口，执行前设置：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
```

设置后启用逐 MoE 低内存导出；未设置或取假值时走普通导出。CLI 仍默认
`--packed-weight-only`，只监控/使用一张可见 GPU，且默认不传
`--enable-auto-offload`。基于 xhquant HMONNX 的交付约束，低内存路径最终生成的
`prefill.hmonnx`、`decode.hmonnx` 必须与同代码、同 checkpoint、同配置的普通
导出逐字节一致（MD5 一致）；placeholder 全部消除、逐 tensor 的量化权重/scale/
zero-point 一致以及同输入数值一致只作为差异定位手段，不能替代这个硬门禁。

持久 Cache 的 meta tensor 只用于避免在导出机上真实分配 256 Ki backing，
不代表运行时 Cache 被迁移到 Host。导出期模型按显式 CPU conversion 流程加载，
但不会启用自动 device-map eviction 或磁盘 offload；默认策略仍是“不 offload”。

### 14.4 融合 mHC 后的真实 first6 导出和 packed demo

AutoRound first6 在 fused `MHCSinkhorn` 路径上的完整低内存导出耗时约 15 分钟，
最终审计如下：

| 项目 | prefill | decode |
|---|---:|---:|
| 节点 | 4,476 | 4,464 |
| `MHCSinkhorn` | 12 | 12 |
| fused `RMSNorm` | 50 | 50 |
| `Rope` / `MoeBlock` / FP16 `Clip` | 26 / 12 / 12 | 26 / 12 / 12 |
| `Cast` / `ConstantOfShape` | 0 / 0 | 0 / 0 |
| ONNX 主文件 | 14,829,826 bytes | 14,822,949 bytes |
| external data | 41,381,618,688 bytes | 41,381,544,192 bytes |

真实 HMONNXInferenceV2 首次加载还执行了全量运行时门禁：36 个 W4 图引用全部
packed，归并为 prefill/decode 共用的 18 份唯一 W4 tensor；source 约 36 GiB，
runtime packed 约 18 GiB，未开启 auto-offload。短链路已经实际完成一次
prefill-256 和一次 decode-1，状态输出提交为 2/2，并成功生成 token。

这次真实运行同时修复了三个不能只靠图审计发现的全链路问题：

- HMONNX 是静态 dtype 图，demo 只迁移 device，不再调用不受支持的
  `model.to(dtype=...)`；
- tokenizer 的 INT64 `input_ids` 在 Host processor 统一成 INT32，图中仍保持
  `Cast=0`；
- MoE parser 不再提前清空共享 initializer 的 `values`；initializer 的生命周期由
  runtime 的 last-use 分析统一管理，因此多个 `MoeBlock` 可以复用同一 packed
  tensor；
- export 和 demo 都在导入重模块前默认设置
  `TORCHINDUCTOR_COMPILE_THREADS=1`，避免 CUDA 初始化后启动 32 个 fork worker
  造成死锁；调用方仍可显式覆盖。

正式 demo 门禁固定为 8,192-token 输入、32 次 prefill-256、生成 512 个新 token、
511 次 decode-1，并要求每次图调用都原位提交 CSA/HCA overlap state。first6 长链路
已经实际通过：`prefill_calls=32`、`decode_calls=511`、`state_commits=543`、
`generated_tokens=512`，且全部持久 cache/state 地址在 543 次图调用前后保持不变。
完整 43 层模型上的同一门禁结果见 14.6。

完整模型不能把四卡 `device_map` 静默截成第一张卡，也不能把“多卡常驻模型并行”
混同为 CPU/磁盘 auto-offload。运行时现在把两者分开：`enable_model_parallel=True`
只在显式 CUDA 列表上做静态层放置，参数始终驻留对应 GPU；
`enable_auto_offload=False` 仍是 release gate。first6 三卡实测还增加了两层放置后审计：

- 18 份 packed W4 在 prefill/decode 中均为 18 份 storage，交集 18、并集 18，
  19,327,352,832 bytes 只驻留一次；
- 所有大于等于 1 MiB 的共享 initializer 共 163 份，prefill/decode 的 storage
  集合完全相同，合计 21,948,434,432 bytes，没有跨卡隐式副本；
- 34 个 first6 SWA/CSA/HCA cache/state 输入按实际消费者分布到三卡
  `13/15/6`，prefill/decode 对每个输入的设备完全一致；连续两次 prefill-256
  加一次 decode-1 的真实运行通过。

最后一项不能省略：若所有 Cache 都先分配到首卡，再由解释器临时 `.to()` 到后续
stage，`LLMCache` 原位更新的只是临时副本，下一 chunk 会读回旧 Cache。当前 runtime
在模型构造时逐个比对 prefill/decode cache placeholder 的消费者设备，并直接在该卡
初始化对应 SWA/CSA/HCA backing 与 compressor state；任何跨图不一致或非 CUDA
落点都会在生成开始前报错。

### 14.5 完整 43 层最终导出

完整 43 层 AutoRound packed artifact 已完成低内存导出。主图复用了 43 对已经导出的
MoE 子图，prefill/decode 各找到并内联 43 个 placeholder；两张图各去重 458 个 inline
initializer，最终图中不再保留 `PlaceHolder`，临时子图目录也已删除。主图导出阶段
prefill/decode 分别耗时约 15 分 02 秒和 14 分 48 秒。

full43 暴露出的导出性能根因不在 MoE 子图，而在 ONNXScript 默认 rewrite：
`_collapse_slices` 会在完整图上长时间组合 Slice 链，`SlicesSplit` 还会两两枚举 Slice
候选；后者在 first6 最终图上单条规则实测即耗时 8.51 秒且没有产生改写。xhquant
现在通过下面两个独立开关精确过滤这三条 Slice 规则，其他 42 条默认规则继续执行：

```bash
export XHQUANT_ONNXSCRIPT_COLLAPSE_SLICES=0
export XHQUANT_ONNXSCRIPT_SLICES_SPLIT=0
```

同时修复了 `optimize_v2` 丢弃 ONNXScript 返回 `ModelProto` 的问题。双过滤后，full43
主图在 constant folding 之后约 30 秒即可完成 ONNXScript optimization；旧路径运行
超过 40 分钟仍不能完成。过滤器只在一次 optimizer 调用期间修改规则集合，退出后
恢复全局规则，且分别有“只移除目标规则并恢复”的定向测试。

最终产物位于：

```text
/data01/home/yujy/work/xh2modelzoo/work_dirs/
  deepseek-v4-flash-full43-autoround-fusedmhc-v33/
  hmquant_deepseek-v4-flash-0731_20260813/
```

确定性序列化与严格结构审计结果如下：

| 项目 | prefill-256 | decode-1 |
|---|---:|---:|
| ONNX 节点 | 44,643 | 44,519 |
| initializer | 3,376 | 3,366 |
| value info | 45,187 | 45,053 |
| ONNX 主文件 | 141,275,489 bytes | 141,216,564 bytes |
| external data | 292,776,094,720 bytes | 292,776,020,224 bytes |
| SHA256 | `37aa177de1bb5e0c1be4bba70df6a453ff7696a385b2dd49dabc2562fa2ae2ed` | `0b9a48af84272d3522eb03b2601d456a3b74b84122b0564a7c09427e583dc` |
| fused `RMSNorm` | 365 | 365 |
| `MHCSinkhorn` | 86 | 86 |
| `Rope` | 212 | 212 |
| `MoeBlock` | 86 | 86 |
| FP16 `Clip` | 86 | 86 |
| Host position inputs | 10 | 10 |
| SWA Cache inputs | 86 | 86 |
| `Cast` / `ConstantOfShape` | 0 / 0 | 0 / 0 |

严格审计还会拒绝残留 placeholder、`Where`/比较型位置逻辑、错误的 MoE routing
输入以及错误的 `Clip` 上下界；因此这里不是只统计期望算子数量，而是完整 ABI 门禁。

完整产物的首次 runtime load 也已通过 packed/shared 门禁。两张图各含 86 个
`MoeBlock`、129 个实际 W4 initializer；单图 W4 source 为 277,025,390,592 bytes，
打包后为 138,512,695,296 bytes。最终运行时统计为：

- `runtime_w4_planned_references=258`；
- `runtime_w4_packed_references=258`；
- `runtime_w4_packed_initializers=129`；
- `runtime_w4_shared_references=129`；
- 唯一运行时 initializer 总量 154,331,193,520 bytes；
- W4 打包节省 138,512,695,296 bytes；
- prefill/decode 共享节省 154,195,608,028 bytes。

也就是每个 W4 tensor 只打包一次，prefill/decode 各引用一次同一对象。首次运行还会
按 dtype、shape、字节数和完整 SHA256 验证两图 initializer 相等，本次 5,186 个
external candidate 的 hash + 打包共耗时 859.823 秒；结果已原子持久化为约 689 KiB
的 digest cache，后续重试不会重新扫描两份约 293 GB external data。

本次最终定向回归结果：GPTQModel DeepSeek-V4 测试 25 passed（2 warnings）；Merak
`tests/deepseek_v4` 为 142 passed（2 warnings）；xhquant 的低内存 loader、W4
runtime contract、模型并行、共享 MoE 常量、量化元数据、`MHCSinkhorn` 和
ONNXScript 过滤器集合为 57 passed；三仓相关 Ruff 检查全部通过。CSA 构造器的
position bias 也改为确定性的零初始化，checkpoint 绑定时再载入真实参数，避免
测试/空壳图中未初始化内存把所有有效 candidate 压到有限哨兵以下。

### 14.6 完整 43 层 8K 输入 / 512-token demo

最终 full43 HMONNX 在物理 GPU 4/5/6/7 上以四卡常驻模型并行运行，进程内映射为
`cuda:0/1/2/3`；`enable_auto_offload=False`，没有 CPU 或磁盘换入换出。正式 release
gate 已实际执行完成并返回 `status=PASS`：

| 门禁 | 实测 |
|---|---:|
| 模型层数 | 43 |
| 输入 token | 8,192 |
| prefill 图长度 / 调用数 | 256 / 32 |
| decode 图长度 / 调用数 | 1 / 511 |
| 新生成 token | 512 |
| 图调用总数 | 543 |
| state commit | 543 |
| 图运行总时间 | 3,154.642 s |
| 持久 Cache/state tensor | 313 |
| Cache/state 设备分布 | 73 / 75 / 75 / 90 |
| Cache consumer-device mismatch | 0 |

第 1 个新 token 直接来自第 32 次 prefill 的最后 logits，因此生成 512 token 只需要
511 次 decode-1；这也是 `32 + 511 = 543` 的原因。脚本同时强制
`min_new_tokens=max_new_tokens=512`，不会因 EOS 提前结束，并在完成后确认所有持久
SWA/CSA/HCA Cache 与 overlap state 的 storage 地址在 543 次调用前后完全不变。

放置后的共享审计也通过：prefill/decode 各见 129 个 packed W4 storage，交集与并集
均为 129；全部大尺寸共享 initializer 在两图中各见 1,182 个 storage，交集与并集
也均为 1,182，总物理共享量 154,021,824,512 bytes。它证明 decode 图没有因为独立
parser 或四卡放置而复制第二份权重。

可复核产物：

```text
work_dirs/deepseek-v4-flash-full43-autoround-fusedmhc-v33/
  demo-8k-512-result.json
  demo-8k-512-console.log
  demo-8k-512-xhquant.log
```

结果 JSON SHA256：
`f49b3bef172f38f1489478341f336ec5399d758b31cb05ecd92253b41a8ddbbe`。

这里必须把“部署链路正确”和“量化质量合格”分开。本次 full43 checkpoint 是用于打通
导出/加载/推理链路的 AutoRound `n1/s16/i1` calibration smoke；512 个输出均为 token
`106618`（解码为重复的 `codeline`），因此不能作为正常语言质量或最终 W4 精度的
验收。上述 PASS 只证明静态图、W4 packing、两图共享、四卡常驻、Cache 更新和精确
512-token 调度完整可运行。正式发布仍需使用完整 calibration 重新量化，并通过 PPL、
KL、下游任务和与原始模型逐层数值对比；不能把本次重复输出写成“质量无损”。

## 15. 下一步验收顺序

1. 使用正式 calibration 配置分别生成 GPTQ 与 AutoRound full43 checkpoint；保持单卡
   block-wise 量化、默认无 offload，并重复当前 packed/共享/8K+512 release gate。
2. 补齐真实尺寸 Block 0～5 的 prefill256、decode1 golden：hidden、mHC、q、
   SWA、compressed KV、Indexer TopK、attention output、MoE output。
3. 将当前 FP16 分离 K/V Cache lower 为 K-SEFP/V-SEFP MXINT8 block64，分别
   验证 scale 归约轴；语义图不再改回共享物理 Cache。
4. 做 78、2359、43456 token 连续 chunk 运行时测试，并检查 HMONNX state
   output 原位回灌和每次 write start。
5. 开始原生 mixed / Baicai W4A16 / GPTQ / AutoRound 的模型级质量评测，至少包含
   PPL、logits KL、任务指标和代表性长文本生成；对 mHC 的任何少步 fast solver 也必须
   在相同输入下与 exact-20 FP16 路径逐层比较后才能启用。

正式放行前必须同时满足：

- 结构：43 层、MTP/DSpark 边界明确；
- 数值：first6 与全模型 golden；
- 量化：PPL/KL/task 指标，不只看单 tensor SNR；
- 性能：Indexer 不物化 2 GiB score，HCA 不复制完整 K/V；
- Cache：K/V 物理分离后的容量、scale、更新和 mask 全部通过长上下文测试。
