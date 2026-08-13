# DeepSeek-V4-Flash CSA 静态图设计

## 1. 设计结论

DeepSeek-V4-Flash 的 CSA（Compressed Sparse Attention）由三条同时工作的路径组成：

1. SWA 保存最近 128 个 token，保证局部信息精确可见；
2. 主 Compressor 每 4 个 token 生成一个 512 维压缩 latent，供真正的 Attention 使用；
3. Indexer 使用另一套 128 维压缩 key，对每个 query token 从完整压缩历史中独立选择 Top-512。

因此，一个 CSA 层不是只有“一份压缩 cache”。当前静态 ABI 中共有三类 cache：

- SWA：物理分开的 `swa_k`、`swa_v`；
- 主压缩历史：物理分开的 `main_k`、`main_v`；
- Indexer：只保存用于检索的 `index_k`。

主压缩 latent 在数学上同时充当 K 和 V，但目标芯片上 K-SEFP 与 V-SEFP 的 scale 归约方向不同，所以必须保留两份物理 cache；Indexer 的 key 仅用于 TopK，不参与最终 value 聚合。

## 2. 固定配置和容量

当前 xh2modelzoo 的发布配置为：

| 项目 | 数值 |
|---|---:|
| Batch | 1 |
| hidden size | 4096 |
| Attention heads | 64 |
| latent head dim | 512 |
| partial RoPE dim | 64 |
| Indexer heads | 64 |
| Indexer head dim | 128 |
| CSA 压缩比 | 4 |
| Indexer TopK | 512 |
| SWA window | 128 |
| 固定 prefill | 256 |
| 固定 decode | 1 |
| 最大上下文 | 256K = 262144 |

最大压缩 cache 长度在 wrap 时确定：

\[
C_{CSA}=\frac{262144}{4}=65536.
\]

每个 CSA 层持久化的张量为：

| 张量 | 形状 | 用途 |
|---|---|---|
| `swa_k` | `[1,1,384,512]` | 最近窗口的 K backing |
| `swa_v` | `[1,1,384,512]` | 最近窗口的 V backing |
| `main_k` | `[1,1,65536,512]` | 主 Attention 压缩 K |
| `main_v` | `[1,1,65536,512]` | 主 Attention 压缩 V |
| `index_k` | `[1,1,65536,128]` | Indexer 压缩 key |

SWA backing 是 `window + prefill = 128 + 256 = 384`。prefill 实际注意力宽度为 `127 + 256 = 383`，向 16 对齐后仍使用 384；decode 只读取最近 128 行。

当前 runtime 实际仍以 FP16 分配这些 cache。`cache_abi.py` 已描述未来 MXINT8 block-64 的 K/V scale 方向，但量化存储尚未真正接入 `LLMCache`，不能把它当成已经完成的 MXINT8 cache。

## 3. 主 Compressor：4 个 token 如何变成 1 个 512 维 latent

### 3.1 两组投影

输入 hidden state 为：

\[
X\in\mathbb{R}^{B\times P\times4096}.
\]

主 Compressor 有两组 `4096 → 1024` 投影：

\[
U=W_{kv}X,\qquad G=W_{gate}X,
\]

其中：

\[
U,G\in\mathbb{R}^{B\times P\times1024}.
\]

1024 维不是最终 latent，而是两个 512 维分支拼在一起：

- 前 512 维服务于相邻窗口的前一组；
- 后 512 维服务于当前窗口。

Gate 还会按 token 在 4-token group 中的位置加 learned position bias：

\[
G'_{b,t,:}=G_{b,t,:}+APE_{t\bmod4,:},
\qquad APE\in\mathbb{R}^{4\times1024}.
\]

`phase_indices=(past_length+arange(P)) mod 4` 由 Host 直接生成并作为图输入，因此图内不再出现位置取模、比较或 `Where`。

### 3.2 为什么一个输出要看 8 个候选

将对齐后的投影 reshape 为：

```text
aligned_u     [B, N+1, 4, 1024]
aligned_gate  [B, N+1, 4, 1024]
```

其中 prefill 的 `P=256`、`N=ceil(256/4)=64`，所以形状为 `[B,65,4,1024]`。

不用构造复杂索引，只需做两次连续 slice，再在候选维拼接：

```text
previous branch = aligned[:, :-1, :, :512]   [B,N,4,512]
current branch  = aligned[:, 1:,  :, 512:]   [B,N,4,512]

candidate = concat(previous branch, current branch, dim=2)
          = [B,N,8,512]
```

Gate 完全相同地构造为 `[B,N,8,512]`。Softmax 在候选维 `dim=2` 上执行，而且每个 feature 独立归一化：

\[
\alpha_{b,n,r,d}
=\frac{\exp(g_{b,n,r,d})}
{\sum_{j=0}^{7}\exp(g_{b,n,j,d})},
\]

\[
c_{b,n,d}
=\sum_{r=0}^{7}\alpha_{b,n,r,d}z_{b,n,r,d}.
\]

形状变化为：

```text
candidate KV/Gate  [B,N,8,512]
softmax             [B,N,8,512]  # dim=2
weighted sum        [B,N,512]     # reduce dim=2
```

这不是 4 个 token 求平均：同一个压缩条目的第 0 维可以主要选择候选 2，而第 300 维可以主要选择候选 6；选择权重由 learned Gate 决定，并且前后两个 4-token group 有 overlap。

### 3.3 RMSNorm 与 partial RoPE

加权结果先做 512 维 RMSNorm，再只对最后 64 维执行 interleaved partial RoPE：

```text
pooled [B,N,512]
  → RMSNorm(512)
  → trailing 64-d partial RoPE
  → compressed latent [B,N,512]
```

压缩位置使用 group 的起始绝对位置：`0,4,8,...`。静态图预先构建压缩 RoPE 表，并以 `csa_write_start=past_length//4` 做一次 `DynamicSlice`，不在图中重新计算位置。

## 4. overlap 状态为什么是 8 行，而不是 4 行

跨 prefill chunk 和 decode 调用时，下一次压缩既需要当前未完成的 group，也需要它前面的完整 group。因此主 Compressor 要保存最近 8 个 token 的两种投影状态：

| 状态 | 形状 | 初值 |
|---|---|---|
| `main_kv_state` | `[1,8,1024]` | 0 |
| `main_score_state` | `[1,8,1024]` | -65504 |

Indexer 有独立的压缩权重，所以还要保存：

| 状态 | 形状 | 初值 |
|---|---|---|
| `index_kv_state` | `[1,8,256]` | 0 |
| `index_score_state` | `[1,8,256]` | -65504 |

这里使用 FP16 有限最小值 `-65504`，图中不允许出现 `-inf`。第一次调用时 Host 传入 `offset=4`，有效输出一定包含当前真实 group；固定图额外产生的无效输出又会被 `compressor_validity` 清零且不会写 cache，因此初始无效状态不会变成有效压缩项。

每次调用的状态更新是固定形状的：

```text
raw_state = concat(previous_8_rows, current_fixed_input, dim=1)
next_state = DynamicSlice(raw_state, start=current_length, length=8)
```

这个 slice 恰好保留“截至本次真实输入末尾的最后 8 行”：

- prefill 不足 256 时不会把右侧 pad 写入状态；
- decode=1 时等价于丢弃最老 1 行，再加入最新 1 行。

状态是普通显式图输入/输出；大容量 K/V cache 则由 `LLMCache(inplace=True)` 原地更新，不作为巨大的图输出回传。

## 5. Host 负责的压缩状态机

设本次调用前已有 `past_length` 个真实 token，本次有 `current_length` 个真实 token：

\[
pending=past\_length\bmod4,
\]

\[
new\_count=\left\lfloor\frac{pending+current\_length}{4}\right\rfloor,
\]

\[
offset=4-pending,
\]

\[
write\_start=\left\lfloor\frac{past\_length}{4}\right\rfloor.
\]

Host 为两个静态图分别准备：

| 输入 | prefill 形状 | decode 形状 | 含义 |
|---|---|---|---|
| `csa_compressor_validity` | `[64]` | `[1]` | 前 `new_count` 个压缩输出有效 |
| `csa_compressor_new_count` | `[1]` | `[1]` | 本次真正写入几行 |
| `csa_compressor_offset` | `[1]` | `[1]` | 固定 candidate slice 的起点 |
| `csa_compressor_phase_indices` | `[256]` | `[1]` | 每个输入 token 的 APE phase |
| `csa_write_start` | `[1]` | `[1]` | 三份长 cache 的写入起点 |

offset 不需要作为模型状态输出；下一次调用直接由新的绝对 `past_length` 重新计算。这避免了图内 `Sub/Mod/Less/Where`，也消除了 offset 与 cache 长度不同步的可能。

## 6. 三份压缩 cache 如何并行更新

主 Compressor 和 Indexer Compressor 接收相同的 Host transition，但使用不同权重和独立 overlap state：

```text
hidden_states
  ├─ main compressor    → [B,N,512] → main_k/main_v
  └─ index compressor   → [B,N,128] → H128 → index_k
```

三次写入共享：

- 同一个 `write_start`；
- 同一个 `new_count`；
- 同一组逻辑压缩位置。

但三份 cache 不共享物理内存：

```text
main_k  [1,1,65536,512]
main_v  [1,1,65536,512]
index_k [1,1,65536,128]
```

当 decode 的 `new_count=0` 时，三次 writer 都必须是空写，长 cache 完全不变，但四份 8-row overlap state 仍然更新；当第四个 token 到来使 `new_count=1` 时，三份 cache 在同一逻辑位置各写入一行。

## 7. Indexer 的完整计算

### 7.1 Indexer key

Indexer Compressor 将 hidden state 投影为两路 128 维分支，按照与主 Compressor 相同的 8-candidate 公式生成：

```text
compressed index key [1,N,128]
  → partial RoPE(last 64 dims)
  → H128/sqrt(128) FP16 MatMul
  → index_k cache [1,65536,128]
```

Hadamard 矩阵是一个固定 FP16 initializer，只做一次矩阵乘，不再展开成蝶形 `Gather/Add/Sub/Concat`。

### 7.2 Indexer query 与 head weight

当前 token 的 query residual 为 `[1,P,1024]`：

```text
q_residual
  → q_b_proj(1024 → 64×128)
  → reshape [1,P,64,128]
  → partial RoPE(last 64 dims)
  → H128/sqrt(128)
  → query [1,P,64,128]

hidden_states
  → weights_proj(4096 → 64)
  → index_weights [1,P,64]
```

query 和 index key 同时经过正交 Hadamard，因此在不量化的实数计算下不改变点积；它主要用于改善后续低精度表示的分布。

### 7.3 每个 token 独立打分并 Top-512

完整容量上的逐 head 点积为：

```text
query       [1,P,64,128]
index_k     [1,65536,128]
    ↓ MatMul
score_head  [1,P,64,65536]
```

然后执行：

\[
s_{b,p,c}=\frac{1}{\sqrt{64\times128}}
\sum_{h=1}^{64}
w_{b,p,h}\operatorname{ReLU}
\left(q_{b,p,h,:}\cdot k_{b,c,:}\right).
\]

形状为：

```text
ReLU(score_head) × index_weights[...,None]
  → ReduceSum(head axis)
  → score [1,P,65536]
  → TopK(K=512, axis=-1)
  → values/indices [1,P,512]
```

所以每个 query token 的 TopK 都可以不同；不能先用一个 `[1,65536,128]` 全局 TopK 再让 256 个 token 共享结果。当前静态图直接保留完整 workspace，并且每个 CSA 层只有一个全局 TopK，不再使用旧实现的 64 个 tiled TopK 加 63 次 merge TopK。

prefill `P=256` 时，`score_head` 有 1,073,741,824 个 FP16 元素，workspace 约 2 GiB；decode `P=1` 时约 8 MiB。该方案计算直接、图结构清晰，但对芯片 workspace 是明确的性能约束。

### 7.4 无效 cache 区域如何处理

对绝对 query 位置 `t=past_length+p`，只允许看已经完整压缩的 group：

\[
valid\_count(t)=\left\lfloor\frac{t+1}{4}\right\rfloor.
\]

Host 生成：

```text
csa_index_validity [1,P,65536]  # FP16 1/0
```

图内只用有限值算术：

\[
s'=s\cdot validity+(-65504)\cdot(1-validity).
\]

因此未写入的 cache 行不会正常进入 TopK。固定 Top-512 在早期上下文仍会返回 512 个 index，但后续 Attention mask 只开放前 `min(valid_count,512)` 个已选 slot。

严格来说，如果某个有效分数恰好饱和到 FP16 最小值 `-65504`，它可能与无效项打平；这是有限 sentinel 方案的边界风险，需要在真实 Indexer score 分布中统计。当前图禁止 Inf/NaN，因此不能靠 `-inf` 消除这个理论 tie。

## 8. TopK 后如何进入真正的 Attention

TopK index 为：

```text
indices [1,P,512]
```

主 K/V cache 使用同一组 index 分别直接 Gather：

```text
main_k [1,65536,512] + indices → selected_k [1,P,512,512]
main_v [1,65536,512] + indices → selected_v [1,P,512,512]
```

必须直接按 cache row Gather。旧实现曾先把 cache Expand 成 `[1,P,65536,512]`，prefill 时单份 K 或 V 就需要 16 GiB；当前实现只物化最终 `[1,256,512,512]`，单份约 128 MiB，并由 HMONNX audit 明确禁止容量级 Expand。

最终 Attention 同时看 Top-512 压缩历史与 SWA：

| 阶段 | TopK K/V | SWA K/V | score/mask 宽度 |
|---|---|---|---:|
| prefill | `[1,256,512,512]` | `[1,384,512]` | `512+384=896` |
| decode | `[1,1,512,512]` | `[1,128,512]` | `512+128=640` |

计算过程为：

```text
compressed_scores [1,64,P,512]
swa_scores        [1,64,P,T_swa]
  → concat
  → add Host csa_attention_mask [1,1,P,512+T_swa]
  → SinksSoftmax（每个 head 另有一个 sink logit）
  → compressed value 聚合 + SWA value 聚合
  → [1,P,64,512]
```

sink 由 `SinksSoftmax` 单独传入，所以显式 Attention score/mask 的最后一维是 896 或 640，而不是再加 1。

当前图的拼接顺序是“压缩 TopK 在前、SWA 在后”。官方参考代码先构造 window index，再拼 compressed index；两者在实数语义上只是同一集合的排列，但低精度 Softmax 的归约顺序可能产生舍入差，因此逐层数值验证时也要把这一点单独统计。

## 9. 不同输入长度的状态转移示例

### 9.1 78-token prompt

输入右 pad 到 256，但 Host 传入 `current_length=78`：

```text
past=0, pending=0, offset=4
new_count=floor(78/4)=19
write_start=0
```

只向三份压缩 cache 写入 19 行；第 20～64 个固定输出被 validity 清零且不提交。下一状态保存真实 token 70～77 对应的最后 8 行，pad 不进入状态。调用结束后 `past=78`、`pending=2`。

继续 decode 时：

```text
处理位置78：past=78, pending=2, new_count=0
处理位置79：past=79, pending=3, new_count=1, write_start=19
```

位置 79 到来后才完成 group 76～79，并写入第 20 个压缩条目。

### 9.2 256-token prompt

```text
past=0, current=256
offset=4, new_count=64, write_start=0
```

压缩条目数正好为 64。下一次 256-token chunk：

```text
past=256, current=256
offset=4, new_count=64, write_start=64
```

虽然 chunk 边界与 group 边界对齐，仍保留上一 chunk 最后 8 行，因为新 chunk 的第一个 overlap 输出需要上一组与当前组的两个分支。

### 9.3 2359-token prompt

静态调度为 10 次调用：

```text
9 × 256 + 55 = 2359
```

前 9 次各写 64 行，共 576 行；最后一次：

```text
past=2304, current=55, pending=0
new_count=floor(55/4)=13
write_start=576
```

最终压缩 cache 有：

\[
576+13=589=\left\lfloor\frac{2359}{4}\right\rfloor
\]

个有效条目，并留下 3 个未完成 token 的 phase 状态。

### 9.4 43456-token prompt

```text
169 × 256 + 192 = 43456
```

共 170 次 prefill 调用：

```text
前169次：169 × 64 = 10816 rows
最后192 token：192 / 4 = 48 rows
总计：10864 rows
```

与 `43456/4=10864` 完全一致。

### 9.5 256K 最大上下文

```text
262144 / 256 = 1024 prefill calls
262144 / 4   = 65536 compressed rows
```

最后一次写入恰好填满预分配的 CSA cache，不需要动态扩容。

## 10. prefill 与 decode 共用的静态 ABI

两个图共享同一份持久 cache 和权重，区别仅在固定 query 长度：

| 项目 | prefill 图 | decode 图 |
|---|---:|---:|
| input length | 256 | 1 |
| 最大新压缩行 | 64 | 1 |
| compressor candidate groups | 65 | 2 |
| candidate tensor | `[1,64,8,D]` | `[1,1,8,D]` |
| Indexer query | `[1,256,64,128]` | `[1,1,64,128]` |
| TopK output | `[1,256,512]` | `[1,1,512]` |
| SWA attention width | 384 | 128 |
| CSA attention width | 896 | 640 |

decode 每 4 步只有 1 步产生新压缩行，其余 3 步是 `valid_count=0` 的空 cache 写入；但 Indexer 每一步都要对新的 query 重新计算 Top-512。

## 11. 当前实现与官方动态实现的差异

| 项目 | 官方参考实现 | 当前 xh2 静态图 |
|---|---|---|
| Compressor 精度 | 显式 FP32 | 纯 FP16 |
| cache 长度 | 随有效上下文增长 | wrap 时固定 65536 |
| TopK 大小 | `min(512, available)` | 固定 512 + Host mask |
| 位置/phase | Python 动态状态 | Host 显式输入 |
| cache 更新 | Python buffer 赋值 | `LLMCache(inplace=True)` |
| 无效值 | `-inf` | FP16 finite min `-65504` |
| Hadamard | 激活旋转/低精度模拟 | FP16 常量 MatMul |
| 主 latent K/V | 数学上共享 | 为不同 SEFP scale 方向物理分开 |
| Attention 拼接顺序 | window 后接 compressed | compressed 后接 SWA |

这些差异中，固定容量、Host mask 和物理 K/V 分离是部署表达方式变化；Compressor 从 FP32 改为 FP16 则是数值变化，必须通过逐层压缩输出、TopK 一致率、hidden state 和长文本指标验证，不能只凭结构审计判定等价。

## 12. 已完成的图结构门禁

当前完整 43 层模型含 21 个 CSA 层，prefill/decode 每张图都满足：

- 每个 CSA 层只有 1 个直接 TopK，共 21 个；
- TopK 的 `K=512`，输出 shape 分别为 `[1,256,512]`、`[1,1,512]`；
- 每个 CSA 层有 3 个长 cache writer；
- 主 K/V 通过 direct Gather 取得，不存在 `[1,P,65536,512]` 容量级 Expand；
- phase、validity、write count、write start 和 Attention mask 全部来自 Host；
- 图中不依赖 `Inf/NaN`，mask sentinel 为 `-65504`；
- 不使用图内 `Mod/Less/Where` 重建压缩控制流。

## 13. 仍需完成的验证和优化

1. **FP16 数值对齐**：对官方 FP32 Compressor 做逐层 `compressed latent / index key / TopK / hidden state` 对比，统计误差和 TopK overlap。
2. **真实长文本阶梯验证（已完成）**：第一组开放许可连续中文经 tokenizer 得到 13305 token，256、1K、2K、4K、8K 五档图执行均通过；8K/max512 在生成 241 token 时正常 EOS，32 次 prefill、240 次 decode、272 次状态提交完全匹配。第二组采用 Project Gutenberg 公版《三国演义》连续正文（562626 token，未机械重复），16K/32K/64K/128K 分别精确执行 64/128/256/512 次 prefill，cache 状态提交、固定 storage、有限值和显存均通过，128K 三卡显存高水位约 71.6/70.7/74.5 GiB。
3. **语义门禁问题（已修复）**：首轮泛化概括问题中，16K、32K 都生成了相同的 19-token 通用拒答，但旧门禁只检查重复 token，错误记录为 `LADDER_PASS`；现已新增 `generic_refusal_template` 判定，并让 ladder 即使某档失败也继续收集后续档。针对正文中明确存在的董卓、曹操、刘备重新做 8K/16K/32K 检索测试，三档均生成对应人物行动且无拒答，说明旧现象不是 CSA/cache 的长度结构性故障，但旧首轮 JSON 只能证明图执行通过，不能作为四档语义全通过证据。
4. **MXINT8 cache 落地**：将 K 的 feature-axis block-64 scale、V 的 token-axis block-64 scale 真正接入 cache writer/gather/Attention；当前仅有 ABI 描述。
5. **finite sentinel tie 审计**：确认真实有效 Indexer score 不会饱和到 `-65504`，否则需要额外的有效项优先级设计。
6. **workspace 评估**：direct TopK 的 prefill head-score workspace 约 2 GiB；当前选择是以显存换取简单稳定的图，后续若芯片 workspace 紧张再设计融合 score-reduce-TopK 算子。
7. **空写边界**：芯片 HMONNX 路径必须继续覆盖 decode `new_count=0/1`，确认空写不改长 cache、有效写只提交一行。

## 14. 对应实现

- `xhmodel_merak/xh_llm/models/deepseek_v4/static_cache.py`：overlap transition、cache writer、direct TopK、direct K/V Gather；
- `xhmodel_merak/xh_llm/models/deepseek_v4/compressor.py`：主/Indexer learned Compressor；
- `xhmodel_merak/xh_llm/models/deepseek_v4/indexer.py`：Indexer query、H128、TopK；
- `xhmodel_merak/xh_llm/models/deepseek_v4/static_attention.py`：SWA、Compressor、Indexer 与最终 Attention 串联；
- `xhmodel_merak/xh_llm/models/deepseek_v4/host_masks.py`：Host phase、validity 和 mask；
- `xhmodel_merak/xh_llm/models/deepseek_v4/runtime.py`：持久 cache、状态回灌和静态输入预处理；
- `xhmodel_merak/xh_llm/models/deepseek_v4/hmonnx_audit.py`：最终静态图结构门禁。
