# DeepSeek-V4-Flash 在 xh2modelzoo 的静态图适配设计与实测

> 更新日期：2026-08-13
> 目标模型：`DeepSeek-V4-Flash-0731`
> 目标部署：XH2a，固定 Prefill 256、Decode 1、最大上下文 256K
> 权重方案：普通线性层 W8、非共享 routed experts W4、group size 64、对称量化
> 部署浮点约束：图上只允许有限 FP16，不允许 FP32、NaN、Inf

## 1. 当前结论

当前适配已经完成并实测了以下基础链路：

1. repaired AutoRound checkpoint 可以按混合 W4/W8 规则被 xh2modelzoo 的 streamed loader 正确识别。
2. 768 个 routed expert Linear/层按 W4 处理，3 个 shared expert Linear/层和 4 个 attention Linear/层按 W8 处理。
3. mHC 的 20 轮 Sinkhorn 已封装成一个 `MHCSinkhorn` 大算子；不再把几十次归一化展开到 ONNX 主图。
4. CSA Indexer 已改为完整 65536 cache 上的直接 score 和单次 TopK-512，不再使用 64 个局部 TopK 加树形 merge。
5. compressor offset、phase、new count、validity、attention mask、cache write start 都由 Host 生成。
6. H128 已明确为归一化 FP16 常量矩阵；最终应走 FP16 MatMul，不能被 frontend 偷换成 W8 Linear。
7. first-6 repaired-v2 v37 packed export 与 HMONNX V2 单卡 load-only gate 均已成功；旧的节点爆炸和两图权重重复驻留均已消除。
8. SWA/CSA/HCA 的静态状态转移在 CPU 参考测试中通过；78、2359、43456 和 256K 边界未发现 off-by-one。
9. first-6 v37 已真实执行 256-token Prefill + Decode-1；两次状态 commit 完整，Decode 的 CSA/HCA `valid_count=0` long-cache 写入路径通过。

尚未完成、不能提前宣称完成的部分：

- 完整 43 层 repaired-v2 HMONNX 正在导出；8K + Decode 512 demo 尚未完成；
- 芯片上的 `LLMCache(valid_count=0)` 边界验证；
- 真正的 MXINT8/SEFP K/V cache。当前物理 K/V 已分离，但实际 cache tensor 仍是 FP16；
- mHC FP16 与官方 FP32 的整网 PPL 等价验证；
- 真实激活下 CSA TopK 集合一致率和 H128 FP16/SEFP16 选择。

## 2. AutoRound 权重与加载状态

### 2.1 混合量化范围

43 层 checkpoint 中共有 33,325 个 packed QuantLinear：

- W4 routed experts：33,024 个；
- W8 attention/shared expert：301 个。

每层对应：

- 256 个 routed experts × `gate/up/down` = 768 个 W4；
- shared expert `gate/up/down` = 3 个 W8；
- attention `q_a/q_b/kv/o_b` = 4 个 W8。

前三个 hash-MoE block 也包含 routed experts，因此它们的 expert 权重仍然走 W4；“hash”只改变路由方法，不改变专家权重位宽。

### 2.2 已修复的两个全局加载根因

第一处根因是 Transformers 5.13 的 `DeepseekV4Experts` 没有被旧 Defuser 路径识别，导致 33,024 个 numbered expert checkpoint module 被 `module_tree` 排除。现在只有当 checkpoint 确实包含 numbered experts、HF tree 又缺失时，才调用 GPTQModel 官方的 MoE defuse 准备逻辑。

第二处根因是 GPTQ v1→v2 qzero 转换错误使用全局 W8 `bits=8`。对于 W4 expert，这会把 packed zero 按错误粒度加偏移，最终产生非法的 `+8` signed code。现在转换使用每个物理 QuantLinear 自己的 `bits` 和 `pack_dtype`，不能再在后续 converter 中统一补 `+1`。

真实 repaired-v2 layer-0 probe 结果：

- 3085 个 buffer 全匹配；
- 771 个 packed Linear 全部转换；
- 768 个 W4 experts 和 3 个 W8 shared Linears 规则正确；
- 加载、wrap 成功。

### 2.3 first-N 低内存导出的全局修复

旧逻辑从 full-43 checkpoint 导出 first-6 时，只跳过前 6 层 MLP，反而仍预处理：

\[
37\times775 + 6\times4 = 28699
\]

个不需要的 packed shell，并真实物化 layer 6–42 的 attention。这不是日志噪声，而是低内存导出作用域错误。

现在 preprocess/defuse/replace 使用统一的 `include_module_prefixes + skip_module_prefixes` 作用域：

- first-6 placeholder template：4626 个 shell，即 `6 × 771`；
- first-6 main graph：24 个 shell，即 `6 × 4`；
- 不再出现 layer ≥ 6 的 materialize/load；
- full-43 保持 33,325 个，不改变完整模型行为。

修复后 first-6 主模型 wrap 前 RSS 从约 15.6 GiB 降到约 5.9 GiB。

## 3. mHC

### 3.1 数学语义

模型使用 `hc_mult=4`、`sinkhorn_iters=20`。兼容模式必须复现官方固定 20 轮，而不是把它替换为“更快求同一个极限”的 Newton/Birkhoff 投影。

当前 fused `MHCSinkhorn` 保留：

- FP16 输入输出；
- `iters=20`；
- `eps=1e-6`；
- 官方 row/column normalization 顺序；
- 每个 mHC mix 一个不透明大算子节点。

### 3.2 导出收益

first-6 旧展开图：

- Prefill 5940 个节点；
- Decode 5928 个节点。

仅融合 mHC 后的旧 tiled-TopK 图：

- Prefill 4476；
- Decode 4464；
- 每张图减少 1464 个节点。

加入 direct CSA TopK 后，first-6 repaired-v2 主图在 MoE 内联前降到：

- Prefill 1460；
- Decode 同量级；
- 主图 Prefill 导出约 46 秒，不再卡在 ONNXScript。

因此，mHC 大算子确实解决了导出慢的主要问题之一；CSA TopK 重构又进一步消除了另一组重复子图。

### 3.3 数值验收要求

融合只解决图规模，不自动证明 FP16 数值与官方 FP32 等价。最终必须用同一真实 `mix logits` 比较：

- fused 与展开 20 轮的逐 token 输出；
- 每层 hidden-state cosine/relative RMSE；
- 相同数据集上的 PPL；
- 最终生成质量。

在这些数据出来前，低步数 Newton 只能作为实验模式，不能替换兼容默认路径。

## 4. 统一静态图规格

### 4.1 固定调用

- Prefill graph：`P=256`；
- Decode graph：`P=1`；
- 最大上下文：`262144`；
- 不足 256 的 Prefill chunk 在右侧 pad；
- `current_length` 表示当前 chunk 的有效 token 数；
- 所有 padding 只占静态槽位，不得写入有效 cache。

请求分块示例：

| 输入长度 | Prefill 调用 | 最后一块有效长度 |
|---:|---:|---:|
| 78 | 1 | 78 |
| 2359 | 10 | 55 |
| 43456 | 170 | 192 |
| 262144 | 1024 | 256 |

### 4.2 Host 输入

主图现在包含：

- `past_seq_length`；
- `current_input_length`；
- `last_token_index=current_input_length-1`；
- `csa_write_start=floor(past/4)`；
- `hca_write_start=floor(past/128)`；
- SWA/CSA/HCA attention masks；
- CSA Indexer validity；
- CSA/HCA compressor validity、new count、offset、phase indices；
- 每层固定 cache/state tensors。

`last_token_index` 也由 Host 直接传入，避免网络内再做 `current_length - 1` 的整数 Sub。网络不应出现为 Host 已知坐标服务的 Cast、Reshape、Less、Where、Mod 或 Sub。

## 5. SWA

window 为 128，Prefill chunk 为 256。物理 backing 不需要 512：

\[
(window-1)+P=127+256=383
\]

按 16 对齐后是 384。

| 项目 | Prefill | Decode |
|---|---:|---:|
| 物理 K/V backing | `[1,1,384,512]` | 同一 backing |
| attention view | `[1,384,512]` | `[1,128,512]` |
| score | `[1,64,256,384]` | `[1,64,1,128]` |

Host mask 控制：

- 每个 query 只能看当前及之前 token；
- 最多保留最近 128 个 token；
- pad token 完全不可见。

当前 K/V 已物理分开，为后续 K 沿 feature、V 沿 token 的不同 scale 方向保留了必要条件。

## 6. CSA 主 Compressor

### 6.1 Host 相位和写入数

对历史有效长度 `Lpast`、当前有效长度 `Lcur`、ratio=4：

\[
p=L_{past}\bmod4
\]

\[
n_{new}=\left\lfloor\frac{p+L_{cur}}4\right\rfloor
\]

\[
offset=4-p
\]

注意对齐状态的 offset 是 4，不是 0。

每个固定 256 槽位的 phase：

\[
phase[i]=(p+i)\bmod4
\]

这些量全部由 Host 计算。图中：

- `phase_indices` 只进入 APE Gather；
- `offset` 只进入 DynamicSlice；
- `new_count` 只控制 cache writer；
- validity 只把无效 pooled 行清零。

### 6.2 overlap 状态与候选生成

主 compressor 投影宽度为 `2×512=1024`，保留 8 行 KV state 和 8 行 gate state。

Prefill 256 时：

```text
history state [B,8,1024]
current       [B,256,1024]
cat raw       [B,264,1024]
DynamicSlice(offset, 260 rows)
reshape       [B,65,4,1024]
```

然后不需要复制完整 tensor，只需两次相邻 group slice：

```text
left  = aligned[:, :-1, :, :512]   # [B,64,4,512]
right = aligned[:, 1:, :, 512:]    # [B,64,4,512]
cat candidate dim                   # [B,64,8,512]
```

gate 使用同样布局，在 8 这个维度执行 softmax：

\[
w[b,g,r,d]=\operatorname{softmax}_{r=0\ldots7}(s[b,g,r,d])
\]

\[
c[b,g,d]=\sum_{r=0}^{7}w[b,g,r,d]\,z[b,g,r,d]
\]

也就是每个 512 特征独立在 8 个候选 token 上归一化，不是 4-token 平均，也不是整行共享一个 gate。

下一状态从 `raw[current_length:current_length+8]` 取 8 行，因此同一个静态表达同时适用于 Prefill 和 Decode。

### 6.3 `-65504` 的使用

初始 score state 使用 `-65504`，KV state 使用 0。第一次 Prefill 的 offset=4，左侧无效历史不会进入有效候选，因此不会出现“8 个全无效却 softmax 成均匀分布并写入 cache”的问题。

仍需保持两个约束：

- `-65504` 只能作为有限无效哨兵；
- 无效 pooled 行必须再乘 Host validity，不能仅依赖 softmax 哨兵。

## 7. CSA Indexer

### 7.1 H128

Indexer query 和 compressed key 都乘：

\[
H_{128}/\sqrt{128}
\]

H128 是固定 FP16 initializer，每个 CSA 层共享同一个矩阵，由 query 和 key 各消费一次。

已发现 xhquant frontend 原先会把“MatMul 的第二输入为二维 get_attr 常量”自动改写成 `nn.Linear`，从而把 H128 送入 W8 权重量化。这不符合设计。现在 `xhnn.MatMul` 可以显式设置 `preserve_constant_rhs`，H128 保留为 FP16 MatMul operand。

first-6 小图验收：

- 2 个 CSA 层；
- 每层 query/key 两次；
- 共 4 个 H128 MatMul；
- RHS shape 均为 `[128,128]`；
- RHS ONNX dtype 为 FLOAT16；
- 不存在 hadamard Linear。

如果后续真实 TopK 一致率不够，再比较 SEFP16；不能在没有数据前把常量静默量化成 W8。

### 7.2 直接 full-cache TopK

当前计算为：

```text
query       [1,P,64,128]
key_cache   [1,65536,128]
MatMul
score_head  [1,P,64,65536]
ReLU × head_weight
ReduceSum(head=64)
score       [1,P,65536]
TopK(512)
indices     [1,P,512]
```

每个 token 的 query 不同，因此每个 token 都有自己的 TopK；不是用一个 `[1,65536,128]` 结果给所有 token 共用。

Prefill 最大 `score_head`：

\[
1\times256\times64\times65536\times2\text{ bytes}=2\text{ GiB}
\]

Decode 为 8 MiB。用户确认显存足够，因此采用这个最直观的静态表达。

Host 为每个 query 构造有效 compressed prefix：

\[
N_q=\left\lfloor\frac{L_{past}+q+1}{4}\right\rfloor
\]

即使当前 chunk 的 64 个 compressed rows 已一次写入完整 cache，早期 query 仍只能看到自己的 prefix，不会偷看未来压缩条目。

无效 score 使用有限值 `-65504`，TopK 后 attention mask 再屏蔽补入的无效 index。

### 7.3 节点下降

旧 tiled 方案每个 CSA 层包含 64 次 local TopK 和 63 次 merge TopK；first-6 每张图共 257 个 TopK。

direct 方案：

- CSA TopK：2 个；
- first-6 主图总 TopK 由正常 MoE router 和这 2 个 CSA TopK 组成；
- MatMul 从旧图 148 降到约 26，其中新增 4 个是明确保留的 H128 FP16 MatMul；
- 不再有 tiled merge 引入的大量 Add/Sub/Concat/Gather/Reshape。

## 8. HCA

HCA ratio 为 128，cache capacity：

\[
262144/128=2048
\]

状态为：

- KV `[B,128,512]`；
- gate `[B,128,512]`。

Host 公式：

\[
n_{new}=\left\lfloor
\frac{L_{past}\bmod128+L_{cur}}{128}
\right\rfloor
\]

Prefill 256 最多输出 2 行；Decode 静态保留 1 行输出槽，但 127/128 次调用的 `new_count=0`。

HCA attention：

| 项目 | Prefill | Decode |
|---|---:|---:|
| long K/V | `[1,2048,512]` | 同 |
| long score | `[1,64,256,2048]` | `[1,64,1,2048]` |
| 拼 SWA 后宽度 | 2432 | 2176 |

## 9. 最大上下文和边界状态

| 长度 | CSA 已写 | CSA 余数/下次 offset | HCA 已写 | HCA 余数/下次 offset |
|---:|---:|---:|---:|---:|
| 78 | 19 | 2 / 2 | 0 | 78 / 50 |
| 2359 | 589 | 3 / 1 | 18 | 55 / 73 |
| 43456 | 10864 | 0 / 4 | 339 | 64 / 64 |
| 262144 | 65536 | 0 / 4 | 2048 | 0 / 128 |

256K 最后一块：

- CSA 写 `[65472,65536)`；
- HCA 写 `[2046,2048)`；
- 两者恰好占满，不越界也不少分配。

## 10. Cache 布局与当前真实状态

### 10.1 物理 shape

每层按类型：

```text
SWA:
  swa_k, swa_v

CSA:
  swa_k, swa_v
  csa_k, csa_v
  csa_index_k
  main kv/gate state
  index kv/gate state

HCA:
  swa_k, swa_v
  hca_k, hca_v
  main kv/gate state
```

K/V 必须分开，因为目标 scale 方向不同：

- K：沿 head feature 维 block64；
- V：沿 token 维 block64；
- Indexer K：沿 128 feature 维 block64。

### 10.2 当前并没有真正使用 MXINT8 cache

目前 `cache_abi.py` 已描述上述 scale 方向，但 runtime 真正分配的仍是 FP16 `CacheTensor`；`LLMCache` 没有接收 INT8 value、FP16 scale、scale axis 或 block64 参数。

因此当前准确表述是：

- K/V 分离和 shape 设计正确；
- MXINT8/SEFP cache 的 value+scale 存储尚未落地；
- SWA rolling 时 token-axis V scale 的同步滚动也尚未实现。

估算 43 层单请求 persistent cache：

- 当前 FP16：约 3.06 GiB；
- 未来 MXINT8 value + FP16 block64 scale：约 1.58 GiB。

## 11. HMONNX 图审计

最新有效产物是 repaired-v2 first-6 v37：

`/data01/home/yujy/work/xh2modelzoo/work_dirs/deepseek-v4-flash-first6-autoround-fp16-directcsa-repaired-v37`

| 指标 | Prefill | Decode |
|---|---:|---:|
| 节点 | 1549 | 1537 |
| 图输入 | 53 | 53 |
| RMSNorm | 50 | 50 |
| mHC input RMSNorm | 13 | 13 |
| MHCSinkhorn | 12 | 12 |
| Rope | 26 | 26 |
| MoeBlock | 12 | 12 |
| FP16 Clip | 12 | 12 |
| MatMul | 26 | 26 |
| CSA direct TopK | 2 | 2 |
| long-cache input/writer | 10 / 10 | 10 / 10 |
| compressor state input | 12 | 12 |
| Cast | 0 | 0 |
| ConstantOfShape | 0 | 0 |

所有图可见浮点 tensor/initializer/Constant 均要求：

- dtype = FP16；
- finite；
- 无 NaN/Inf；
- Clip 使用 `±65504` 等有限边界。

v37 中两层 CSA 的 query/key Hadamard 共对应每张图 4 个 MatMul；每个右操作数都是 `[128,128]` 的 ONNX FP16 initializer，没有被 frontend 改成 W8 Linear。两个 CSA TopK 的真实输入分别为：

- Prefill：`[1,256,65536]`；
- Decode：`[1,1,65536]`。

其余 3 个 TopK 是 MoE 的 256-expert 路由，不是 CSA merge 子图。`last_token_index` 只有一个直接 Gather 消费者，`current_length` 没有 Sub 消费者。两图内嵌浮点常量全部 finite。

v37 总导出时间 586.68 秒，CPU peak RSS 约 38.19 GiB，GPU peak memory 约 2.63 GiB。Prefill/Decode external data 分别约 41.38 GB；此前 v35/v36 和其他旧 first-6 产物均已删除，不能再作为有效结果引用。

### 11.1 packed W4 与两图共享的真实加载结果

v37 已使用 HMONNXInferenceV2 在单张 A100 上完成 `load-only` gate，未启用 offload：

- W4 planned/packed references：`36 / 36`；
- unique/shared W4 initializers：`18 / 18`；
- packed W4 实际唯一驻留：19,327,352,832 bytes；
- Prefill/Decode packed W4 storage：`18 / 18`，共享 storage：18；
- 两图所有 material shared storage：`163 / 163`，地址完全共享；
- cache/state tensor：34 个，全部驻留 `cuda:0`，mismatch 为 0；
- 状态：`LOAD_ONLY_PASS`。

第一次加载因建立 704 项 initializer SHA256 cache 用时约 104.9 秒；缓存命中后第二次相同共享分析降到约 20.2 秒。单卡默认 legacy runtime 无法执行 packed residency audit，因此 DeepSeek-V4 release gate 已改为无条件选择 HMONNXInferenceV2，不再依赖容易漏传的 `--use-v2`。

### 11.2 first-6 HMONNX 真实执行 smoke

在另一张空闲 A100 上执行了 256-token Prefill 一次和 Decode-1 一次：

- `prefill_calls=1`；
- `decode_calls=1`；
- `state_commits=2`；
- cache/state storage 调用前后地址稳定；
- 图执行总时间约 26.01 秒；
- 状态：`PASS`。

该 Decode 步从 `past_length=256` 开始，CSA ratio=4 和 HCA ratio=128 都没有形成新的完整压缩条目，因此真实覆盖了两类 long-cache writer 的 `valid_count=0` 路径。HMONNXInferenceV2 执行完成且固定容量 cache 未被破坏。生成文本不用于模型质量判断，因为该产物只有前 6 个 block；文本质量必须以 full-43 的 8K/512 release gate 为准。

上述最终 audit 已覆盖：

- CSA/HCA cache capacity 和 state shape；
- 每个 CSA 层恰好一个 direct TopK；
- writer 数量和 long-cache K/V 分离；
- `last_token_index` 直接进入 Gather；
- `current_length` 不再进入 Sub。

## 12. Demo 和验收计划

### 12.1 图结构 gate

1. full-43 Prefill/Decode 均无 placeholder；
2. 每层 mixed W4/W8 placeholder 内联完整；
3. 43 层 mHC 数量正确；
4. CSA/HCA/SWA cache 和 state ABI shape 正确；
5. H128 是 FP16 constant MatMul；
6. graph-visible 浮点全部 FP16 finite；
7. Cast/ConstantOfShape/位置比较和坐标重建算子为 0。

### 12.2 Runtime gate

必须开启：

- runtime W4 packing；
- Prefill/Decode initializer sharing；
- no offload；
- aligned cache 路径，不能开启会返回动态 prefix 的 `--fast`。

必须验证：

- Prefill/Decode W4 只保留一份 GPU storage；
- cache/state 地址在调用间稳定；
- CSA/HCA state 输出逐层回灌；
- `valid_count=0` 时 cache bitwise 不变；
- `valid_count=1` 时只写一行。

### 12.3 数值和生成 gate

1. 先做短 prompt smoke；
2. 512-token one-shot 与 2×256 chunk 对照；
3. 78、2359、43456 chunk 边界对照；
4. 精确 8192-token prompt，32 次 Prefill-256；
5. `max_new_tokens=512`，逐 token Decode-1；
6. 检查 finite logits、重复模式、EOS、吞吐；
7. 比较官方/Transformers 与 HMONNX 的 PPL、Top-1、KL、hidden state；
8. 单独报告 FP16 GEMM 因 M=512/M=256 reduction 顺序不同造成的正常舍入差，不能把它误判为 cache 位置错误。

## 13. 当前优先级

1. 完成正在运行的 repaired-v2 full-43 导出和最终图审计；
2. full-43 做 8 卡、无 offload、packed/shared runtime load；
3. 跑精确 8K + Decode 512 demo并定位任何阶段性错误；
4. 做 HMONNX 与 GPTQModel/Transformers 的 chunk、PPL、Top-1、KL 数值对照；
5. 再落地 MXINT8 K/V value+scale cache；
6. 用数据决定 H128 FP16 是否需要切换 SEFP16，以及 mHC 是否可以提供低迭代 fast mode。
