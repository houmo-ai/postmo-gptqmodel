# TensorRT-LLM 适配 DeepSeek-V4-Flash-0731：前 6 个 Block、Prefill-256、Decode-1 与 256K 容量设计

> - 文档日期：2026-08-11
> - 目标 checkpoint：`/data01/datasets/DeepSeek-V4-Flash-0731`
> - TensorRT-LLM 参考版本：`59e8079ab09423d4c700724fdd80ce45d8c76042`（2026-08-10，`main`）
> - 分析范围：主模型 Block 0～5、Prefill-256、Decode-1、256K 静态 Cache 容量与多 chunk 输入
> - 补充范围：xh2modelzoo 当前 mHC / CSA / HCA 实现、Cache 数据流与长上下文缺口
> - 说明：本文使用全局逻辑 shape，不展开 TP/EP/PP 分片后的局部 shape。

## 1. 先给结论

Block 0～5 恰好覆盖 DeepSeek-V4-Flash 主干中的全部 decoder layer 类型：

| Block | checkpoint ratio | TensorRT-LLM 内部 ratio | Attention | MoE 路由 |
|---:|---:|---:|---|---|
| 0 | 0 | 1 | SWA，仅最近 128 个 raw KV | Hash Top-6 |
| 1 | 0 | 1 | SWA，仅最近 128 个 raw KV | Hash Top-6 |
| 2 | 4 | 4 | CSA：SWA-128 + 压缩历史 Top-512 | Hash Top-6 |
| 3 | 128 | 128 | HCA：SWA-128 + 全部 128× 压缩历史 | Learned Top-6 |
| 4 | 4 | 4 | CSA：SWA-128 + 压缩历史 Top-512 | Learned Top-6 |
| 5 | 128 | 128 | HCA：SWA-128 + 全部 128× 压缩历史 | Learned Top-6 |

其中：

- CSA 是 Compressed Sparse Attention；
- HCA 是 Heavily Compressed Attention；
- 每层都有两套 mHC，分别包住 Attention 和 MoE；
- 每个 token 同时执行 6 个 routed expert 和 1 个 shared expert；
- 前 3 层并不是 Dense FFN，而是用 token-ID hash 表决定 6 个 routed expert；
- Block 3 开始换成可学习的 `sqrt(softplus(logit))` router。
- 最大上下文按 256×1024=262,144 token 计算时，每个 CSA 层需要 65,536 条 main/Indexer compressed Cache，每个 HCA 层需要 2,048 条 compressed Cache；
- HCA 的固定 Attention index width 是 2,176，但 Prefill 图只需保留当前 256 个 query 的索引 workspace，不需要常驻整个 256K query×index 矩阵。

TensorRT-LLM 的核心解法不是在 PyTorch 图中物理执行：

```text
concat(SWA_cache, compressed_cache) -> 普通 Attention
```

而是：

```text
两个独立的 paged cache pool
  + 每个 query 的固定宽度索引表
  + 每个 query 的有效索引长度 topk_lens
  + 一个支持双池的 sparse MLA/FMHA kernel
```

这个 kernel 先处理 SWA pool，再处理 compressed pool，并在一次 online-softmax 中合并两部分。因此 Cache 不需要真的 concat，动态长度也被藏在索引、长度和自定义 kernel 内。

还要先说明：TensorRT-LLM 本身使用 packed/ragged token，不会把 78 个真实 token 当成 256 个 token 写 Cache。本文的“Prefill 固定 256”是把 TensorRT-LLM 的语义映射成自研芯片的静态图接口；PAD token 必须通过 `valid_len` 和 write mask 完全隔离。

## 2. 本模型的关键配置

本地 `config.json` 的主干配置为：

| 项目 | 数值 |
|---|---:|
| hidden size | 4096 |
| decoder layers | 43 |
| attention heads | 64 |
| KV heads | 1 |
| head dim | 512 |
| Q LoRA rank | 1024 |
| KV latent / NoPE dim | 448 |
| partial-RoPE dim | 64 |
| output groups | 8 |
| output LoRA rank | 1024 |
| SWA window | 128 |
| CSA compression ratio | 4 |
| HCA compression ratio | 128 |
| Indexer heads | 64 |
| Indexer head dim | 128 |
| Indexer Top-K | 512 |
| routed experts | 256 |
| shared experts | 1 |
| activated routed experts | 6 |
| expert intermediate dim | 2048 |
| hash-routing layers | 3，即 Block 0～2 |
| mHC multiplicity | 4 |
| Sinkhorn iterations | 20 |

checkpoint 的 `compress_ratios` 以 `[0, 0, 4, 128, 4, 128, ...]` 开头。TensorRT-LLM 在 API/Cache 内部把 `0` 规范化为 `1`，只为避免除零并统一 Cache 数学；语义仍然是 SWA-only，不会真的做 1× Compressor。

### 2.1 本文 shape 记号

以单请求、一个完整 256-token chunk 为例：

```text
B = 1
L = 256
N = B × L = 256
H = 4096
HC = 4
NH = 64
D = 512 = 448 NoPE + 64 RoPE
DI = 128 = 64 NoPE + 64 RoPE
```

Decode 第一步：

```text
B = 1
L = 1
N = 1
absolute position p = 256
```

## 3. TensorRT-LLM 从 checkpoint 到可执行模型做了什么

### 3.1 Config 归一化

TensorRT-LLM 为 V4 临时维护 `DeepseekV4Config`，并做如下 alias 映射：

- checkpoint `head_dim=512` 映射到内部 `v_head_dim=512`；
- `qk_nope_head_dim` 缺省为 448；
- `kv_lora_rank` 缺省为 448；
- `sliding_window` 映射成内部 `window_size`；
- `num_hash_layers` 映射成内部 `n_hash_layers`；
- ratio 0 在 sparse runtime 中归一化成 1。

当前实现的主要约束是：

- 仅 PyTorch backend；
- 当前完整 Attention 路径要求 Blackwell，代码中对 pre-SM100 直接报错；
- `window_size` 必须是 128；
- ratio 只允许 1、4、128；
- KV page 的 `tokens_per_block` 只允许 128 或 256；
- `num_kv_heads=1`；
- `max_beam_width=1`。

注意这里有三种完全不同的 “block”，不能混用：

1. Transformer Block，例如 Block 2；
2. KV page 的 `tokens_per_block=128/256`；
3. 权重量化的 block，例如 FP8 128×128、MXFP4 K-block 32、目标芯片 K-block 64。

### 3.2 权重重命名和融合

checkpoint 使用 `layers.i.attn.*`、`layers.i.ffn.*` 和平铺的 `hc_attn_fn` 等名字；TensorRT-LLM 加载时做重命名和结构融合。

主要变换包括：

1. 将 checkpoint 的 `attn.wq_a` 与 `attn.wkv` 在输出维拼接成一次共享 GEMM：

   ```text
   kv_a_proj_with_mqa:
       [N,4096] × [1536,4096]^T -> [N,1536]
                                -> qr [N,1024]
                                 + raw_kv [N,512]
   ```

2. 每个 Compressor 的 `wkv` 和 `wgate` 融合成一个 `wkv_gate`：

   ```text
   ratio4 main:    [N,4096] -> [N,2048]
                   1024 KV + 1024 Gate

   ratio4 indexer: [N,4096] -> [N,512]
                   256 KV + 256 Gate

   ratio128 main:  [N,4096] -> [N,1024]
                   512 KV + 512 Gate
   ```

3. `wo_a` 被整理成 8 组独立的低秩输出投影；
4. flat mHC 权重映射成 `mHC.fn/base/scale`；
5. `attn_sink[64]` 以 FP32 加载，并按 TP head 分片；
6. routed-expert 的 packed MXFP4 与 shared-expert 的 FP8 走不同量化配置；
7. Block 0～2 的 `tid2eid[vocab,6]` hash 表被保留并传给 MoE router。

### 3.3 该 checkpoint 的精度分工

本地 tensor header 验证结果如下：

| 组件 | checkpoint 精度 |
|---|---|
| Attention `wq_a/wq_b/wkv/wo_a/wo_b` | FP8 E4M3，scale 为 UE8M0（PyTorch `float8_e8m0fnu`），二维 block 128×128 |
| Shared expert `w1/w2/w3` | FP8 E4M3，scale 为 UE8M0，二维 block 128×128 |
| Routed expert `w1/w2/w3` | packed MXFP4 E2M1，K-block 32，scale 为 UE8M0 |
| Main/Indexer Compressor `wkv/wgate` | BF16 |
| Indexer `wq_b` | FP8 E4M3，128×128 |
| Indexer `weights_proj` | BF16 |
| Router weight | BF16 |
| mHC `fn/base/scale` | FP32 |
| Attention sink | FP32 |

主 Attention KV Cache 的 dtype 是运行时选择，不由权重 checkpoint 唯一决定：TensorRT-LLM 支持 BF16 或 per-tensor FP8。Indexer-K Cache 在 Blackwell 上默认使用 MXFP4，也可选 blockwise FP8。Compressor 的未完成窗口状态固定保留 FP32。

### 3.4 TensorRT-LLM 的软件接线与实际调用链

V4 并没有另写一套完整 MLA Python 模型，而是把算法挂到公共 `MLA` 上：

```text
DeepseekV4ForCausalLM
  -> DeepseekV4Model
    -> DeepseekV4DecoderLayer
      -> 公共 MLA
        -> 注册名为 "deepseek_v4" 的 DeepSeekV4Hooks
          -> DeepseekV4TrtllmAttention backend
             + Compressor（ratio > 1）
             + DeepseekV4Indexer（仅 ratio = 4）
```

`DeepSeekV4Hooks` 明确关闭 dense-MHA、普通 output projection 和 MLA absorption 路径；它接管四个阶段：初始化 V4 专用模块、分配 Attention 输出、执行 sparse forward、执行 grouped output projection。

`DeepseekV4CacheManager` 为不同时间状态分配 7 种逻辑 role：

| role | 生命周期 | 哪类层使用 | 内容 |
|---|---|---|---|
| `SWA` | sliding | 全部层 | 最近 128 个 raw latent |
| `COMPRESS` | non-sliding | CSA/HCA | 512-d main compressed latent |
| `INDEXER_COMPRESS` | non-sliding | 仅 CSA | 128-d Indexer K |
| `COMPRESSOR_KV` | sliding state | CSA/HCA | main Compressor 未完成窗口的 value projection |
| `COMPRESSOR_SCORE` | sliding state | CSA/HCA | main Compressor 未完成窗口的 Gate score |
| `INDEXER_COMPRESSOR_KV` | sliding state | 仅 CSA | Indexer Compressor 的 value projection |
| `INDEXER_COMPRESSOR_SCORE` | sliding state | 仅 CSA | Indexer Compressor 的 Gate score |

一次 forward 前，metadata 层先根据每个 request 的 `cached_len/new_len` 生成：sliding/compressed block table、每种 ratio 的新压缩条目数、压缩条目的 position、scatter mask、每个 query 的 SWA/HCA local indices，以及 `topk_lens`。模型层不再自己判断 page 地址。

随后 `forward_sparse_attn` 的调度顺序是：

1. ratio>1 时尽早在辅助 stream 发射 main Compressor；
2. 对 hidden state 做融合的 Q-A/KV-A GEMM，再并行执行 Q/KV RMSNorm；
3. 发射 Q-B + per-head norm；CSA 同时执行 Indexer 的 hidden-only prework、Indexer Q/K 和 Top-K；
4. 等待 Compressor/Indexer 的事件，只同步真正被 Attention 消费的结果；
5. Prefill 和 Decode 分别进入 context/generation wrapper，但最终都调用 V4 sparse MLA backend；
6. backend 把 SWA local index 与 compressed local index 通过各自 page table 转成两个 pool 的全局地址，并把 `topk_lens`、sink、Cache 指针交给 sparse MLA/FMHA；
7. 得到 64×512 latent head 输出后，执行 inverse RoPE、8 组 `o_a` 和最终 `o_b`。

这条调用链解释了为何 TensorRT-LLM 能保持 CUDA Graph 外形固定：变化的是 metadata 内容和 paged state，而不是 Python 层 Tensor shape。

## 4. 一个 decoder Block 的公共骨架：mHC → Attention → mHC → MoE

### 4.1 为什么 residual 变成 4 份

Embedding 原始输出为：

```text
E: [N,4096] BF16
```

进入 Block 0 前复制 4 份：

```text
R0 = repeat(E, HC=4): [N,4,4096] BF16
```

这 4 份是 mHC residual streams，不是 attention heads，也不是 4 份 KV Cache。

### 4.2 一次 mHC pre-mapping 的精确计算

对每个 token 独立计算。令：

```text
R      : [4,4096]
r      = flatten(R): [16384]
W_fn   : [24,16384] FP32
base   : [24] FP32
scale  : [3] FP32
```

先计算：

```text
z = W_fn @ r                         # [24]
rho = rsqrt(mean(r^2) + 1e-6)        # scalar
z_hat = z * rho                      # [24]
```

24 个动态参数分成 `4 + 4 + 16`：

```text
a_i = sigmoid(z_hat[i] * scale[0] + base[i]) + eps
      i=0..3                                      # pre-mix [4]

c_j = 2 * sigmoid(z_hat[4+j] * scale[1] + base[4+j])
      j=0..3                                      # post-mix [4]

L_ij = z_hat[8+4i+j] * scale[2] + base[8+4i+j]
      i,j=0..3                                    # comb logits [4,4]
```

`L` 先做 row softmax 并加 `sinkhorn_eps`，接着做一次 column normalization；然后再执行 `sinkhorn_iters-1=19` 轮 row normalization + column normalization，得到近似双随机矩阵。也就是说，配置中的 20 不是“row/column 成对循环 20 次”：

```text
G = Sinkhorn20(L): [4,4]
sum_j G[i,j] ~= 1
sum_i G[i,j] ~= 1
```

Attention 或 MoE 的单流输入为：

```text
u[h] = sum_i a_i * R[i,h]
u: [4096]
```

批量 shape 为：

```text
pre_mix    a: [N,4,1] FP32
post_mix   c: [N,4,1] FP32
comb_mix   G: [N,4,4] FP32
layer_input u: [N,4096] BF16
```

### 4.3 mHC post-mapping

假设子层输出为 `F(u): [4096]`，则写回 4 条 residual stream：

```text
R'[j,h] = c_j * F(u)[h] + sum_i G[i,j] * R[i,h]
```

这就是 V4 的 residual connection。Attention/MoE 外面不能再额外做一次普通 `x + residual`，否则语义错误。

### 4.4 TensorRT-LLM 的 boundary fusion

朴素执行一个 Block 需要：

```text
hc_attn.pre
Attention
hc_attn.post
hc_ffn.pre
MoE
hc_ffn.post
```

TensorRT-LLM 把相邻边界融合：

- Block 内部：`hc_attn.post + hc_ffn.pre` 合为一次 `hc_ffn.fused_hc`；
- 跨 Block：上一层 `hc_ffn.post + 下一层 hc_attn.pre + 下一层 input RMSNorm` 合为下一层入口的 `hc_attn.fused_hc`。

所以执行状态是：

```text
Block 0 entry: standalone hc_attn.pre + RMSNorm
Block 0 exit : defer hc_ffn.post

Block 1 entry: consume Block 0 deferred state，执行 fused_hc + RMSNorm
Block 1 exit : defer hc_ffn.post

...依次类推...
```

`HCState.deferred` 携带：

```text
residual [N,4,4096]
post_mix [N,4,1]
comb_mix [N,4,4]
x_prev   [N,4096]
```

它只是相邻 Block 间的执行中间态，不是跨 Prefill/Decode step 保存的时间 Cache。

如果只单测 Block 0～5 并在 Block 5 截断，必须显式执行 Block 5 的 `hc_ffn.post_mapping`；如果继续执行 Block 6，则让 Block 5 保持 deferred，由 Block 6 入口吸收。

## 5. 三种 Attention 的公共 Q、raw KV 和输出投影

### 5.1 共享 A projection

对 mHC + RMSNorm 后的 `X: [N,4096]`：

```text
[qr, kv_raw] = fused_a(X)

qr     : [N,1024]
kv_raw : [N,512]
```

分别做：

```text
qr = RMSNorm_weighted(qr)        # [N,1024]
kv_raw = RMSNorm_weighted(kv_raw)# [N,512]
```

### 5.2 Query 路径

```text
Q = q_b(qr)                 # [N,32768]
Q = view(Q,[N,64,512])
Q = per-head RMSNorm(Q)     # 无 learnable weight
Q[...,448:512] = RoPE64(Q[...,448:512], absolute_position)
```

### 5.3 raw SWA KV 路径

`kv_raw` 同时作为 K latent 和 V latent：

```text
K/V NoPE : kv_raw[...,0:448]
K/V RoPE : RoPE64(kv_raw[...,448:512], absolute_position)
```

每层把这个 512-d latent 写入自己的 VSWA/paged Cache；逻辑上是 128-token 滑窗，物理页位置由 sliding block table 决定。

对绝对位置 `p`，可见 raw token 是：

```text
[max(0,p-127), ..., p]
```

当前 token 自己包含在 causal Attention 中。

主 Attention 对 raw 与 compressed 两类条目使用同一套 latent 计算。对 head `h`、query `p`、被选中的 memory item `j`：

```text
logit[p,h,j] = dot(Q[p,h,:], K[j,:]) / sqrt(512)
O[p,h,:]     = sum_j softmax(logit)[p,h,j] * V[j,:]
```

这里每个 cache item 的同一条 512-d latent 同时充当 `K` 和 `V`；最后 64 维已经按该 item 的 position 做过 RoPE。Attention 输出的最后 64 维随后执行 inverse RoPE，再进入 grouped output projection。`attn_sink[h]` 只给 softmax 分母增加一个零-value memory item，详见第 8.3 节。

这里的“同一条 latent”只表示数学值共享，不表示自研芯片上可以共享同一份量化 Cache。XH2 的 K-SEFP 沿 head-dim 分组/共享指数，而 V-SEFP 沿 token 轴分组/共享指数；两者的 scale 归约轴、物理排布和更新粒度不同。因此 SWA raw、CSA main compressed、HCA main compressed 都必须各自保留独立的 K Cache 与 V Cache，即使写入前的 BF16 latent 数值完全相同。仅 CSA Lightning Indexer 是 K-only 检索路径，不需要 V Cache。

### 5.4 RoPE 配置

- ratio 1 / checkpoint ratio 0：base RoPE，`theta=10000`，不开 YaRN；
- ratio 4/128：使用 compression RoPE，`theta=160000`，YaRN factor 16；
- 仅最后 64 维参与 RoPE；
- 压缩条目的 RoPE position 是压缩窗口的第一个 raw position `r*j`，不是压缩序号 `j`，也不是窗口完成位置 `r*j+r-1`。

### 5.5 Attention 输出投影

双池 Attention 输出为：

```text
O_attn: [N,64,512]
```

先对最后 64 维做 inverse RoPE，再把 64 个 head 分成 8 组，每组 8 个 head：

```text
[N,64,512]
 -> [N,8,8*512]
 -> [N,8,4096]
```

每组独立执行：

```text
o_a: 4096 -> 1024
```

8 组拼成 `[N,8192]`，最后：

```text
o_b: 8192 -> 4096
```

得到 Attention 子层输出 `[N,4096]`。

## 6. Compressor：怎样并行生成压缩 Cache，又不让 query 看见未来

### 6.1 通用 gated pooling

Compressor 不是平均池化。令 compression ratio 为 `r`，输出 dim 为 `D_c`。

对每个 raw token 投影：

```text
V_t: [D_c]
S_t: [D_c]
P_a: [D_c]，a=t mod r      # learnable APE，实际加到 Gate score
```

非 overlap 的 HCA 对第 `j` 组执行：

```text
group j = raw token [r*j, ..., r*j+r-1]

alpha[j,a,d] = softmax over a=0..r-1 of (S[r*j+a,d] + P[a,d])

C[j,d] = sum_a alpha[j,a,d] * V[r*j+a,d]
```

shape：

```text
V_group     [num_groups,r,D_c]
score_group [num_groups,r,D_c]
softmax dim = r 这个 token-offset 维
```

不同 channel `d` 有不同的 softmax 权重；既不是 128 个 token 求平均，也不是整条向量共用一个权重。

### 6.2 CSA ratio-4 的 overlap 公式

CSA 的 main Compressor 输出 `D_c=512`，但每个 raw token 投影两条 branch：

```text
V^A_t, V^B_t : 各 [512]
S^A_t, S^B_t : 各 [512]
```

因此 checkpoint 中 main Compressor 的 `wkv`、`wgate` 各输出 1024。

对第 `j` 个压缩输出，候选轴长度为 8：

```text
Z[j,0:4,d] = V^A[4*(j-1) : 4*j, d]   # 上一组的 A branch
Z[j,4:8,d] = V^B[4*j : 4*j+4, d]     # 当前组的 B branch
```

Gate 同样取上一组 A branch 和当前组 B branch，并分别加对应的 4-position APE。然后：

```text
alpha[j,:,d] = softmax(G[j,:,d], dim=candidate_axis_8)
C[j,d] = sum_{a=0..7} alpha[j,a,d] * Z[j,a,d]
```

shape：

```text
Z/G   : [num_groups,8,512]
alpha : [num_groups,8,512]
C     : [num_groups,512]
```

`j=0` 没有上一组，前 4 个 value 填 0，前 4 个 Gate 填 `-inf`，所以第一个输出只使用 raw token 0～3 的 B branch。

连续输出的关系是：

```text
C[j]   使用 group(j-1).A + group(j).B
C[j+1] 使用 group(j).A   + group(j+1).B
```

因此每个正常输出具有 8-token receptive field，相邻输出共享中间那组 4 个 raw token，但使用不同 projection branch。

### 6.3 HCA ratio-128

HCA 不 overlap：

```text
r = 128
D_c = 512
V/S group: [num_groups,128,512]
softmax dim: 128-token offset
```

每 128 个 raw token 完成一个 512-d compressed latent，相邻窗口互不重叠。

### 6.4 Indexer Compressor

CSA 还有第二套完全独立的 Compressor，用来生成 Indexer K：

```text
r = 4
D_c = 128
state_dim = 2*128 = 256
输出 K_index_comp: [num_compressed,128]
```

它使用同样的上一组 A + 当前组 B overlap 公式，但拥有自己的 `wkv/wgate/APE/norm`，不能用 main 512-d compressed KV 截断或复用得到。

后处理顺序是：

```text
gated reduction
 -> RMSNorm(128)
 -> last 64 dim partial RoPE，position=4*j
 -> normalized Hadamard transform
 -> MXFP4-B32 或 FP8-B128 quant
 -> INDEXER_COMPRESS paged cache
```

### 6.5 未完成窗口状态

TensorRT-LLM 将每个 raw token 的 `V` 与加过 APE 的 Gate score 以 FP32 保存在短生命周期 paged state 中。

| Compressor | KV state | Score state | 逻辑窗口 |
|---|---|---|---:|
| CSA main | `[8,1024]` FP32 | `[8,1024]` FP32 | 两个 4-token group |
| CSA Indexer | `[8,256]` FP32 | `[8,256]` FP32 | 两个 4-token group |
| HCA main | `[128,512]` FP32 | `[128,512]` FP32 | 一个 128-token group |

CSA 在恰好完成一个 group 后仍必须保留刚完成的 4-token group，因为下一输出还要使用它的 A branch。HCA 在完整 128-token 边界后逻辑 open count 为 0，buffer 中旧值可以留着，下一组会从 slot 0 覆盖。

### 6.6 为什么可以先并行算完整个 256-token chunk

初始 Prefill 256 时，Compressor 可以一次并行得到：

```text
CSA: 256/4   = 64 entries
HCA: 256/128 = 2 entries
```

这些 entry 可以先全部 scatter 到 compressed Cache，但每个 query 的可见上界单独计算：

```text
visible_compressed(p,r) = floor((p+1)/r)
```

例如：

- CSA query p=2：`floor(3/4)=0`，看不到尚未完成的 group 0；
- CSA query p=3：`floor(4/4)=1`，刚好能看见由 token 0～3 完成的 entry 0；
- HCA query p=126：`floor(127/128)=0`；
- HCA query p=127：`floor(128/128)=1`。

所以“先批量生成和写 Cache”不会产生未来信息泄漏，因果性由 per-query compressed bound 保证。Indexer 构造 logits mask 时使用同一个上界。

## 7. CSA Indexer 的详细计算

对 CSA 层，主 Attention 的 `qr: [N,1024]` 同时送给 Indexer Q 路径：

```text
Q_I = indexer.wq_b(qr)          # [N,64*128]
Q_I = view([N,64,128])
Q_I[...,64:128] = RoPE64(..., raw absolute position p)
Q_I = Hadamard(Q_I)
Q_I = quantize to MXFP4/FP8
```

另一条 head-weight 路径：

```text
w = weights_proj(X)             # [N,64] BF16
w *= 1/sqrt(128) * 1/sqrt(64)
```

Indexer Compressor 提供历史 key：

```text
K_I: [T_comp,128]
```

对 query `p`、Indexer head `h`、compressed entry `j`：

```text
d[p,h,j] = dot(Q_I[p,h,:], K_I[j,:])
s[p,j] = sum_h w[p,h] * ReLU(d[p,h,j])
```

只在：

```text
0 <= j < floor((p+1)/4)
```

范围内选 Top-512，得到：

```text
topk_indices: [N,512] int32
```

无效槽填 `-1`。

对初始 256-token Prefill，最多只有 64 个 compressed entry，少于 512，因此数学上 Top-K 等于“全选”。当前 TensorRT-LLM 自动配置允许 short-sequence skip 时，会直接填连续 dense index；若关闭 skip，也可以真的计算 Indexer score 和 Top-K，Attention 结果相同，只是顺序可能不同。真正开始丢弃 compressed entry 要等历史超过 512 个压缩条目，即 raw history 超过约 2048 token。

需要区分：

- `INDEXER_COMPRESS` 是长期保存的 128-d compressed Indexer-K Cache；
- `topk_indices` 是本次 query 的临时选择结果，不是第三份长期 KV Cache；
- GVR 等优化可以保存上一 decode step 的 Top-K 作为 hint，但不改变模型语义。

## 8. 双池 Attention、固定索引和 attention sink

### 8.1 每个 query 的逻辑集合

对绝对位置 `p`：

```text
SWA(p) = raw positions [max(0,p-127), ..., p]
```

三类层分别为：

```text
ratio1:
  K(p) = SWA(p)

ratio4 / CSA:
  K(p) = SWA(p) + Top512(compressed[0 : floor((p+1)/4)])

ratio128 / HCA:
  K(p) = SWA(p) + compressed[0 : floor((p+1)/128)]
```

压缩条目即使覆盖了最近 128 个 raw token，也不会从 compressed pool 中删除。raw 表示和 compressed 表示会作为两个不同 memory item 同时进入 softmax，这是模型定义的一部分，不是重复 Cache bug。

### 8.2 TensorRT-LLM 的固定宽度索引

为了 CUDA Graph shape 稳定：

| ratio | 固定 index buffer 宽度 |
|---:|---:|
| 1 | 128 |
| 4 | `128 + 512 = 640` |
| 128 | `128 + next_pow2(ceil(max_seq_len/128))` |

该 checkpoint 的 `max_position_embeddings=1,048,576`，若 runtime 也按完整 1M 配置，HCA compressed 区固定宽度为 8192，总宽度为 8320。实际 kernel 只读取 `sparse_mla_topk_lens[p]` 指定的有效前缀；无效 index 为 `-1`。

注意 ratio>1 时，SWA 区在 buffer 中永远占满固定 128 个槽。序列开头不足 128 的位置用 `-1`，所以 metadata 的 active span 和真实有效 KV 数不完全相同。

### 8.3 不物理 concat 时怎样精确合并

设 SWA pool 和 compressed pool 分别完成 online-softmax partial reduction：

```text
(m_s, l_s, u_s)  # max、exp sum、未归一化 value sum
(m_c, l_c, u_c)
```

每个 attention head 还有 checkpoint 提供的 sink logit `lambda_h`，sink 对应零 value。精确合并为：

```text
m = max(m_s, m_c, lambda_h)

l = exp(m_s-m)*l_s
  + exp(m_c-m)*l_c
  + exp(lambda_h-m)

u = exp(m_s-m)*u_s + exp(m_c-m)*u_c

O = u / l
```

SWA-only 层令 compressed 部分为空即可。这个公式说明自研静态图也不必构造一个物理 concat Tensor；可以做两个 pool 的 Attention partial，再做稳定 softmax merge。

## 9. MoE 的完整计算

每层 MoE 输入都先经过 `post_attention_layernorm`，shape 为 `[N,4096]`。

### 9.1 Block 0～2：Hash routing

checkpoint 保存：

```text
tid2eid: [129280,6]
```

对 token id `tid`：

```text
expert_ids = tid2eid[tid]       # [6]
```

Hash 决定 expert id，但 router GEMM 仍产生 256 个 FP32 logits，用来计算被选中 6 个 expert 的 routing weight。按官方参考逻辑：

```text
affinity_e = sqrt(softplus(router_logit_e))
weight_k = affinity[selected_k]
weight /= sum(weight)
weight *= 1.5
```

Hash 层没有 score-correction bias。

### 9.2 Block 3 以后：Learned routing

```text
router_logits = X @ W_gate^T       # [N,256] FP32 output
affinity = sqrt(softplus(logits))
rank_score = affinity + correction_bias[256]
expert_ids = TopK(rank_score,6)
```

correction bias 只影响 expert 选择，不进入最终 routing weight：

```text
weight = gather(affinity, expert_ids)
weight = weight / sum(weight) * 1.5
```

### 9.3 Routed 和 shared expert

每个 expert 都是受限 SwiGLU：

```text
gate = W1 x                # [2048]
up   = W3 x                # [2048]

gate = min(gate, 10)
up   = clamp(up, -10, 10)

z = SiLU(gate) * up
y_e = W2 z                 # [4096]
```

最终：

```text
y_routed = sum_{k=0..5} routing_weight[k] * y_{expert_id[k]}
y_shared = SharedExpert(x)
y_moe = y_routed + y_shared
```

TensorRT-LLM 把 routed 和 shared 两条分支放在不同 CUDA stream 上并行，最后相加；EP 部署时 routed experts 还包含 dispatch/all-to-all/combine。

## 10. 初始 Prefill：固定 256 token 的逐步执行

本节假设没有 padding：真实 prompt 恰好是 256 token，绝对位置为 0～255。

### 10.1 Runtime 在进入模型前准备的 metadata

单请求：

```text
cached_raw_len = 0
new_raw_len = 256
total_raw_len = 256
position_ids = [0,1,...,255]
```

各 ratio 的新 compressed 数：

```text
new_comp(r) = floor(total_raw_len/r) - floor(cached_raw_len/r)

ratio4:   64
ratio128: 2
```

compressed RoPE position：

```text
ratio4:   [0,4,8,...,252]
ratio128: [0,128]
```

同时构造：

- 每层 SWA block table；
- ratio4/128 compressed block table；
- ratio4 Indexer-K block table；
- 每个 query 的 SWA local indices；
- HCA 的顺序 compressed indices；
- CSA 的 causal Indexer 上界；
- `sparse_mla_topk_lens`；
- Compressor state 和 compressed output 的 scatter metadata。

若 KV page 设为 128 raw tokens：

```text
CSA compressed page = 128/4   = 32 entries
HCA compressed page = 128/128 = 1 entry
```

若 KV page 设为 256，则分别是 64 和 2 entries。KV page size 与 Prefill chunk=256 只是本例恰好相等，语义上是独立配置。

### 10.2 Embedding 和初始 mHC residual

```text
input_ids          [1,256]
Embedding          [256,4096]
repeat HC=4        [256,4,4096]
```

### 10.3 Block 0：SWA + Hash MoE

1. `hc_attn.pre_mapping`：

   ```text
   residual         [256,4,4096]
   -> attention in  [256,4096]
   ```

2. `input_layernorm`。
3. Fused A projection：

   ```text
   qr     [256,1024]
   raw KV [256,512]
   ```

4. Q projection得到 `[256,64,512]`；对 Q/raw KV 最后 64 维做 base RoPE。
5. 将 raw KV 0～255 写入本层 SWA Cache/scratch；每个 query 只读取自己的 causal 128-window。
6. 双池 Attention 的第二池为空，输出 `[256,4096]`。
7. `hc_ffn.fused_hc` 同时完成 Attention post 和 FFN pre。
8. `post_attention_layernorm`。
9. Hash router 用每个原始 `input_id` 查 6 个 expert，执行 6 routed + 1 shared expert。
10. Block 0 的 `hc_ffn.post` 暂不执行，保存 deferred HCState 给 Block 1。

Prefill 结束时，本层只需长期保留 raw position 128～255；0～127 只在本次 Prefill 计算早期 query 时需要，可走 scratch 生命周期。

### 10.4 Block 1：SWA + Hash MoE

1. Block 1 入口的 `hc_attn.fused_hc` 消费 Block 0 deferred state，同时完成：

   ```text
   Block0 hc_ffn.post
   + Block1 hc_attn.pre
   + Block1 input RMSNorm
   ```

2. Block 1 自己重新生成 Q/raw KV；KV Cache 不跨层共享。
3. 执行 SWA-128 Attention，无 Compressor、无 Indexer。
4. 执行 mid-layer fused mHC。
5. 执行 Hash Top-6 routed experts + shared expert。
6. defer Block 1 `hc_ffn.post` 给 Block 2。

### 10.5 Block 2：CSA + Hash MoE

Block 2 是第一层完整 CSA，也是最后一层 hash-routed MoE。

1. 入口 fused mHC，得到 `[256,4096]` normalized Attention input。
2. TensorRT-LLM 可从同一个 hidden input 提前并行发射三类工作：

   ```text
   A. main Compressor:     生成 64 × 512 compressed attention KV
   B. Indexer prework:     weights_proj + Indexer Compressor，生成 64 × 128 K_I
   C. shared A projection: 生成 qr[256,1024] + raw KV[256,512]
   ```

3. Q 主路径：`qr -> q_b -> [256,64,512] -> per-head RMSNorm -> RoPE`。
4. Indexer Q 路径：`qr -> [256,64,128] -> RoPE -> Hadamard -> quant`。
5. Indexer 对每个 query 只考虑 `floor((p+1)/4)` 个 K_I；本 chunk 最多 64 个，因此可直接 dense fill，也可执行 score + Top-K。
6. 构造双池索引：

   ```text
   前 128 slots: SWA raw indices
   后 512 slots: CSA compressed Top-K indices
   ```

7. dual-pool sparse MLA 做一次联合 softmax，并加入 `attn_sink[64]`。
8. inverse RoPE + grouped `o_a` + `o_b`，输出 `[256,4096]`。
9. mid-layer fused mHC。
10. Hash Top-6 MoE + shared expert。
11. defer FFN post 给 Block 3。

Block 2 的三类长期时间状态是：

```text
SWA raw KV
main compressed KV
Indexer compressed K
```

另外两套 Compressor 的 KV/score state 是短期状态，不是 Attention 的第三、第四个历史池。

### 10.6 Block 3：HCA + Learned MoE

1. 入口 fused mHC。
2. main Compressor 对 raw token 0～127、128～255 分别做 128-token、逐 channel gated pooling，得到：

   ```text
   compressed KV [2,512]
   RoPE positions [0,128]
   ```

3. 同时生成本层 Q `[256,64,512]` 和 raw KV `[256,512]`。
4. HCA 不运行 Indexer，不运行 Top-K；compressed indices 是顺序的：

   ```text
   query p < 127: no compressed entry
   query 127..254: entry 0
   query 255: entries 0,1
   ```

5. dual-pool MLA 合并 SWA 与全部已完成 HCA entries。
6. Attention 输出投影。
7. mid-layer fused mHC。
8. Learned router：sqrt-softplus + correction bias 选 Top-6；同时运行 shared expert。
9. defer FFN post 给 Block 4。

### 10.7 Block 4：CSA + Learned MoE

Attention/Cache/Indexer 流程与 Block 2 相同，区别只有：

- 所有 Q、KV、Compressor、Indexer、sink、Cache 都是 Block 4 自己的参数和存储；
- MoE 改为 learned router；
- Block 4 的 64 个 main compressed entry 与 Block 2 的 64 个 entry 不能共享。

### 10.8 Block 5：HCA + Learned MoE

Attention/Cache 流程与 Block 3 相同，MoE 也是 learned router。

若继续完整模型，Block 5 返回 deferred HCState 给 Block 6；若测试图只截到前 6 个 Block，则在输出前显式 materialize Block 5 `hc_ffn.post_mapping`，输出 `[256,4,4096]` residual state。

### 10.9 256-token Prefill 后每层 Cache 快照

下表按单请求列出逻辑条目；实际显存还会按 page 对齐。

| Block | SWA 最终有效 raw KV | main compressed | Indexer-K compressed | main Compressor state | Indexer state |
|---:|---:|---:|---:|---|---|
| 0 | 128×512 | 无 | 无 | 无 | 无 |
| 1 | 128×512 | 无 | 无 | 无 | 无 |
| 2 | 128×512 | 64×512 | 64×128 | capacity `8×1024` KV + score；最后 4-token group 需保留 | capacity `8×256` KV + score；最后 group 需保留 |
| 3 | 128×512 | 2×512 | 无 | capacity `128×512` KV + score；open count=0 | 无 |
| 4 | 128×512 | 64×512 | 64×128 | 同 Block 2 | 同 Block 2 |
| 5 | 128×512 | 2×512 | 无 | 同 Block 3 | 无 |

所有 Compressor state 均为 FP32。表中的两项 CSA state 都包含 KV state 和同 shape 的 score state。

### 10.10 关键 query 的有效 Attention 数量

这里区分“真实有效 KV 数”和 TensorRT 固定 index buffer 中的 active span。

| query p | SWA 真实数 | CSA compressed 数 | CSA 真实总数 | CSA `topk_lens` | HCA compressed 数 | HCA 真实总数 | HCA `topk_lens` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 1 | 128 | 0 | 1 | 128 |
| 3 | 4 | 1 | 5 | 129 | 0 | 4 | 128 |
| 127 | 128 | 32 | 160 | 160 | 1 | 129 | 129 |
| 255 | 128 | 64 | 192 | 192 | 2 | 130 | 130 |

序列开头 `topk_lens` 看起来比真实 KV 多，是因为固定的 128-slot SWA 区中包含 `-1` padding index，FMHA 会屏蔽它们。

## 11. 紧接 Prefill 的 Decode-1：绝对位置 256

现在 Cache 中已有 raw token 0～255，输入新 token 的绝对位置是 256。

### 11.1 Decode metadata

```text
cached_raw_len = 256
new_raw_len = 1
total_raw_len = 257
```

本步实际新 compressed 数：

```text
ratio4:   floor(257/4)   - floor(256/4)   = 0
ratio128: floor(257/128) - floor(256/128) = 0
```

为了 CUDA Graph shape 固定，TensorRT-LLM decode Compressor 仍可为每个 request 预留一个 output slot；`compressed_mask=False`，所以该 slot 不会 scatter 到 persistent compressed Cache。

### 11.2 Decode 的 mHC shape

每层仍做完全相同的 mHC 数学，只是 token 轴变成 1：

```text
residual   [1,4,4096]
layer input[1,4096]
pre/post   [1,4]
comb       [1,4,4]
```

mHC 没有从 Prefill 保存 per-sequence temporal state；它使用这个新 token 在当前 forward 中逐层传递的 residual state。

### 11.3 Block 0、1

每层：

1. 生成 token 256 的 Q 和 raw KV；
2. raw KV 写入本层 VSWA/paged Cache；position 128 从当前 query 的逻辑窗口退出，窗口更新为 129～256。TensorRT-LLM 通过 sliding block table 映射页面，不能把物理地址简单写成 `256 mod 128`；
3. 当前 query 读取 raw position 129～256，共 128 个；
4. 无 Compressor/Indexer；
5. Block 0/1 仍走 Hash Top-6 MoE。

### 11.4 Block 2、4：CSA

每层：

1. token 256 是新 4-token group 的第 1 个 token；
2. main Compressor 将它的双 branch KV/score 写入 open-window state；
3. Indexer Compressor 同样写入自己的 open-window state；
4. 本步不会生成新的 main compressed entry，也不会生成新的 Indexer-K entry；
5. 历史仍有 64 个 compressed entries，少于 Top-512，因此全部可见；
6. Attention 集合是：

   ```text
   SWA raw positions 129..256: 128
   CSA compressed entries 0..63: 64
   合计真实 192
   ```

7. MoE：Block 2 是 Hash，Block 4 是 Learned。

到 decode position 259，也就是收齐 raw token 256～259 时，才会生成第 65 个 main compressed entry 和第 65 个 Indexer-K entry；该 query 259 可以立即看见新 entry。

### 11.5 Block 3、5：HCA

每层：

1. token 256 写入新 HCA group 的 slot 0；
2. 本步不生成 compressed entry；
3. 仍保留 Prefill 生成的 2 个 entries；
4. Attention 集合：

   ```text
   SWA raw positions 129..256: 128
   HCA entries 0..1: 2
   合计真实 130
   ```

5. 两层均走 Learned MoE。

要等到 decode position 383，即收齐 raw token 256～383，才生成第 3 个 HCA entry。

### 11.6 Decode-1 后的增量 Cache 变化

| Block | SWA | main compressed count | Indexer-K count | open state |
|---:|---|---:|---:|---|
| 0 | 更新为 raw 129～256 | 无 | 无 | 无 |
| 1 | 更新为 raw 129～256 | 无 | 无 | 无 |
| 2 | 更新为 raw 129～256 | 64，不变 | 64，不变 | ratio4 current group 已有 1/4 token，并保留上一组 A branch |
| 3 | 更新为 raw 129～256 | 2，不变 | 无 | ratio128 current group 已有 1/128 token |
| 4 | 同 Block 2 | 64，不变 | 64，不变 | 同 Block 2 |
| 5 | 同 Block 3 | 2，不变 | 无 | 同 Block 3 |

## 12. 多个固定 256-token chunk 怎样连续执行

设第 `k` 个完整 chunk：

```text
start_pos = 256*k
valid_len = 256
```

因为 256 同时能被 4 和 128 整除：

```text
每个完整 chunk 新增 64 个 CSA entry
每个完整 chunk 新增 2 个 HCA entry
```

但不能因为 chunk 对齐就丢弃所有 Compressor state：

- HCA 在边界没有 open group，旧 state 可直接覆盖；
- CSA 的下一个输出还需要上一 chunk 最后一个 4-token group 的 A branch，所以必须跨 chunk 保留最后 4 个 raw token 的 projection state。

例如第二个 chunk 位置 256～511：

- 第一条新 CSA entry 的当前 B branch 来自 256～259；
- 它的上一组 A branch 来自 252～255；
- 后者必须由前一 chunk 的 Compressor state 提供。

对 chunk 内每个 query，causal compressed bound 变成：

```text
floor((absolute_position+1)/r)
```

因此即使第二个 chunk 的 64/2 个新 entries 都先批量 scatter，query 256 仍只读 Prefill 结束前已经完成的 64/2 个 entries。

## 13. 不足 256 的静态 PAD 必须怎样处理

对一个静态 `prefill256` 图，建议显式输入：

```text
input_ids      [B,256]
valid_len      [B]
past_raw_len   [B]
position_ids   [B,256]
```

定义：

```text
valid[b,t] = (t < valid_len[b])
absolute_pos[b,t] = past_raw_len[b] + t
```

PAD lane 可以参与固定 shape 的矩阵计算，但必须满足：

1. 不写 SWA Cache；
2. 不写 Compressor KV/score state；
3. 不触发 group completion；
4. 不增加 compressed count；
5. 不写 main compressed 或 Indexer-K Cache；
6. 不成为任何真实 query 的 K/V；
7. MoE 中不占用会影响真实 token 的 capacity；
8. PAD query 的最终输出和 logits 被丢弃。

通用计数公式是：

```text
new_comp(P,L,r) = floor((P+L)/r) - floor(P/r)
```

其中 `P=past_raw_len`，`L=valid_len`。新 compressed entry 的 RoPE positions 为：

```text
(floor(P/r) + k) * r
```

### 13.1 例：真实长度 78，PAD 到 256

```text
CSA: floor(78/4)   = 19 entries，open remainder=2
HCA: floor(78/128) = 0 entries，open remainder=78
SWA: 仅 78 个真实 raw KV
```

绝不能按 padded length 256 生成 64 个 CSA 和 2 个 HCA entries。

下一 decode token 位置 78 是 CSA 当前 group 的第 3 个 token；到位置 79 才完成 entry 19。HCA 要到位置 127 才完成第一条 entry。

### 13.2 例：真实长度 2359

```text
2359 = 9*256 + 55
```

执行 9 次完整 prefill256，再执行一次 `valid_len=55` 的 padded 图：

```text
CSA total = floor(2359/4)   = 589，remainder=3
HCA total = floor(2359/128) = 18， remainder=55
SWA final = raw positions 2231..2358，共 128 个
```

最后一次 padded 调用只新增：

```text
CSA: floor(2359/4)-floor(2304/4) = 13
HCA: floor(2359/128)-floor(2304/128) = 0
```

### 13.3 例：真实长度 43,456

先拆成固定 256-token 调用：

```text
43,456 = 169 * 256 + 192
```

所以执行：

```text
169 次完整 prefill256
+ 1 次 valid_len=192 的 padded prefill256
```

前 169 次结束时：

```text
past_raw_len = 43,264
CSA count    = 43,264 / 4   = 10,816
HCA count    = 43,264 / 128 = 338
```

最后一次调用覆盖真实 position 43,264～43,455，新增：

```text
CSA new = floor(43,456/4)   - floor(43,264/4)
        = 10,864 - 10,816
        = 48

HCA new = floor(43,456/128) - floor(43,264/128)
        = 339 - 338
        = 1
```

最后一个 chunk 的 compressed RoPE anchor 为：

```text
CSA: 43,264, 43,268, ..., 43,452，共 48 个
HCA: 43,264，共 1 个
```

这一个 HCA entry 覆盖 position 43,264～43,391。之后 position 43,392～43,455 已写入下一 HCA open group，但只有 `64/128`，不能生成第 340 个 entry。

整个 Prompt 完成后的状态为：

```text
past_raw_len       = 43,456
last raw position  = 43,455
SWA                = position 43,328..43,455，共 128 条
CSA main count     = 10,864，remainder=0
CSA Indexer count  = 10,864
HCA main count     = 339，remainder=64
```

CSA 虽然 `remainder=0`，overlap Compressor 仍要保留最后一组 position 43,452～43,455 的 A branch，供下一条压缩输出使用。

最后一个 query `p=43,455` 的实际 Attention 集合是：

```text
CSA: 128 SWA + Top-512 of 10,864 compressed = 640
HCA: 128 SWA + 339 compressed               = 467
```

若整个 runtime 按本文下一节的 256K 上限初始化，则 HCA index buffer 的静态宽度仍是 2176；本 query 只令 `topk_lens=467`，其余位置保持 `-1`，不能因为当前输入只有 43,456 token 而换图或缩 buffer。

此时 CSA Indexer 的最后一个 query 需要在 10,864 个候选中选择 512 个；若图按 256K 最大容量编译，候选轴仍固定为 65,536，通过 `candidate_len=10,864` 屏蔽尾部。按当前长度临时物化 `[256,10864]` FP32 score 约为 10.61 MiB，而固定最大 shape `[256,65536]` 是 64 MiB。

紧接着 Decode-1，输入 position 43,456：

- CSA：进入一个新 4-token group 的第 1 个 token，count 仍为 10,864；到 position 43,459 才生成下一项；
- HCA：已有 64 个 token 的 open group 变为 `65/128`，count 仍为 339；到 position 43,519 才生成下一项；
- SWA：逻辑窗口更新为 position 43,329～43,456。

## 14. 最大 256K 上下文时怎样初始化静态容量

### 14.1 容量口径

本文把 `256K` 按二进制容量解释为：

```text
MAX_SEQ = 256 * 1024 = 262,144 token
合法 position = 0..262,143
```

如果产品定义的 256K 是十进制 256,000，只需将下列公式中的 `MAX_SEQ` 换成 256,000。还要注意，`MAX_SEQ` 应表示 prompt 与后续 decode 合计的最大序列长度；若要允许“完整 256K prompt + 额外输出”，runtime 上限必须再为输出 token 留空间。

固定 Prefill-256 时，一个完整 256K 请求最多执行：

```text
262,144 / 256 = 1,024 次 prefill256
```

对已提交的 persistent compressed Cache，容量公式为：

```text
CSA_CAPACITY = floor(MAX_SEQ / 4)
HCA_CAPACITY = floor(MAX_SEQ / 128)
```

本例刚好整除：

```text
CSA_CAPACITY = 65,536 entries
HCA_CAPACITY = 2,048 entries
```

最后一条 CSA entry 的 RoPE anchor 是 262,140，最后一条 HCA entry 的 anchor 是 262,016、覆盖 position 262,016～262,143。完整写满时 HCA remainder 为 0；CSA 也没有未完成 group，但 overlap state 仍保留最后一个 4-token group 的 A branch。此时已经到达总序列上限，不能再执行 position 262,144 的 Decode，除非预先扩大 `MAX_SEQ`。

到最后一个合法 query `p=262,143`：

```text
CSA Attention = 128 SWA + Top-512 of 65,536 compressed = 640
HCA Attention = 128 SWA + all 2,048 compressed          = 2,176
```

### 14.2 每个请求、每层必须预留的持久状态

以下是逻辑最小 shape，`Bmax` 是静态 batch/request-slot 容量：

| 层类型 | 状态 | 静态容量 |
|---|---|---|
| 全部层 | SWA latent | `[Bmax,128,512]` |
| CSA | main compressed Cache | `[Bmax,65536,512]` |
| CSA | Indexer-K Cache | `[Bmax,65536,128]`，按实际低精度格式打包 |
| CSA | main Compressor KV/score state | KV、score 各 `[Bmax,8,1024]` FP32 |
| CSA | Indexer Compressor KV/score state | KV、score 各 `[Bmax,8,256]` FP32 |
| HCA | main compressed Cache | `[Bmax,2048,512]` |
| HCA | main Compressor KV/score state | KV、score 各 `[Bmax,128,512]` FP32 |

前 6 个 Block 因而可以初始化为：

```text
SWA             [6, Bmax,   128, 512]
CSA main        [2, Bmax, 65536, 512]   # Block 2、4，各自独立
CSA Indexer-K   [2, Bmax, 65536, 128]   # Block 2、4，各自独立
HCA main        [2, Bmax,  2048, 512]   # Block 3、5，各自独立
+ 对应的逐层 Compressor state
```

43 个 backbone Block 中有：

```text
2 个 SWA-only
21 个 CSA
20 个 HCA
```

checkpoint 的 `compress_ratios` 还有末尾 3 个 ratio-0 项，对应 DSpark/MTP 虚拟层，不包含在上述 43 层统计中；启用 DSpark 时要另外计算它们的 SWA 与 speculative rewind 容量。

哪些状态可以跨层共享、哪些不能共享：

- `past_raw_len`、`csa_count=floor(past_raw_len/4)`、`hca_count=floor(past_raw_len/128)` 和 phase 可以按 request 只保存一份；
- 各层的 SWA、compressed Cache、Compressor projection state 必须逐层独立；
- 同一次调用的 `new_csa/new_hca` metadata 应在进入模型前计算一次，供所有同 ratio 层读取；
- 不要让 Block 2 写完后递增一个全局 CSA count，再让 Block 4 读取，否则 Block 4 会错写到下一段地址。

### 14.3 固定索引、page table 与单次调用 workspace

最大 256K 时，三类 Attention 的 index width 固定为：

| 层类型 | 固定 index width |
|---|---:|
| SWA-only | 128 |
| CSA | `128 + 512 = 640` |
| HCA | `128 + next_pow2(ceil(262144/128)) = 128 + 2048 = 2176` |

因此 Prefill-256 图可复用以下 `int32` 索引 workspace：

```text
SWA indices [Bmax,256, 128]   # 每 request 0.125 MiB
CSA indices [Bmax,256, 640]   # 每 request 0.625 MiB
HCA indices [Bmax,256,2176]   # 每 request 2.125 MiB
topk_lens  [Bmax,256]
```

Decode-1 对应每 request 分别只需 512 B、2.5 KiB、8.5 KiB 的 index row。workspace 可以在顺序执行的同类层之间复用，不需要为 21 个 CSA 和 20 个 HCA 各保存一份，也不需要持久保存前 1,023 个 chunk 的 query indices。

Prefill Attention 还需要当前 chunk 的 raw latent scratch：

```text
[Bmax,256,512]
```

它在 BF16 下是每 request 256 KiB，在 FP8 下是 128 KiB；同样只服务当前层/当前 chunk，可按执行调度复用，不能误配成 `[Bmax,262144,512]` 的全历史 raw Cache。

CSA 还有一个更大的候选轴：

```text
Indexer candidate capacity = 65,536
Top-K output width          = 512
```

如果普通静态图把最后一个 Prefill chunk 的 reduced Indexer score 全量物化：

```text
[Bmax,256,65536] FP32 = 每 request 64 MiB
```

逐 head 点积若物化成 `[Bmax,256,64,65536]`，每 request 会达到 4 GiB FP32，实际实现绝不能这样做。推荐 `indexer_score_topk` 在 tile 内完成 64-head reduction，并维护流式 Top-512，只输出 `[Bmax,256,512]`。如果暂时只能用普通 Top-K，至少只物化 head-reduced 的 64 MiB score workspace，并在所有 CSA 层间复用。

每次调用的 Compressor output scratch 上限很小：

```text
prefill256 CSA main       [Bmax,64,512]
prefill256 CSA Indexer-K  [Bmax,64,128]
prefill256 HCA main       [Bmax, 2,512]
decode1                  每种 ratio 预留 1 条并用 compressed_mask 控制提交
```

如果使用 paged Cache，最大 full-history page-table 宽度取决于 raw `tokens_per_block`：

| raw tokens/page | 最大逻辑 pages/request | CSA entries/page | HCA entries/page | Indexer entries/page |
|---:|---:|---:|---:|---:|
| 128 | 2048 | 32 | 1 | 32 |
| 256 | 1024 | 64 | 2 | 64 |

`COMPRESS` 与 `INDEXER_COMPRESS` 的 page table 可以固定为 `[Bmax,2048]` 或 `[Bmax,1024]`。SWA 和 Compressor state 是滑窗状态，不应按 256K 分配 full-history 页；它们只需窗口容量及当前 Prefill chunk 的 scratch。

### 14.4 Cache 容量换算成字节

下面均为单 request、单 layer，且不含 allocator/page-table 开销：

| Cache | 逻辑条目 | FP8 main Cache | BF16 main Cache |
|---|---:|---:|---:|
| SWA 512-d | 128 | 64 KiB | 128 KiB |
| CSA main 512-d | 65,536 | 32 MiB | 64 MiB |
| HCA main 512-d | 2,048 | 1 MiB | 2 MiB |

CSA Indexer-K 单独计算：

```text
MXFP4-B32: 128*0.5 byte + 4*UE8M0 byte = 68 byte/entry
             65,536 entries = 4.25 MiB/CSA layer

FP8-B128:  128 byte + 1*FP32 scale = 132 byte/entry
             65,536 entries = 8.25 MiB/CSA layer
```

Compressor state 的逻辑最小值为：

```text
CSA main:    2 * 8 * 1024 * 4 byte = 64 KiB/layer
CSA Indexer: 2 * 8 *  256 * 4 byte = 16 KiB/layer
HCA main:    2 * 128 * 512 * 4 byte = 512 KiB/layer
```

以“main Cache 使用 FP8、Indexer-K 使用 MXFP4、state 使用 FP32”为例：

| 范围 | 逻辑最小持久容量/request |
|---|---:|
| 前 6 个 Block | 76.03 MiB |
| 43 个 backbone Block | 795.58 MiB |

若 main Cache 全部使用 BF16，两项分别约为 142.41 MiB 和 1,490.27 MiB（约 1.46 GiB）。这些数字不包含当前 chunk activation、Indexer score/Top-K workspace、page table、对齐和 DSpark Cache。

静态 `Bmax>1` 时，上述 per-request 数字近似乘以 `Bmax`；使用 PP 时只计算当前 rank 所持有的 layer，不能一边使用全 43 层数字、一边再次乘 PP rank 数。

若自研方案连 KV Cache 也使用 `MXINT8-B64 + FP16 scale`，则：

```text
512-d latent: 512 byte INT8 + 8*2 byte scale = 528 byte/entry
128-d Indexer: 128 byte INT8 + 2*2 byte scale = 132 byte/entry

CSA main = 65,536 * 528 byte = 33 MiB/layer
HCA main =  2,048 * 528 byte = 1.03125 MiB/layer
Indexer  = 65,536 * 132 byte = 8.25 MiB/CSA layer
```

加上逻辑 FP32 state 后，前 6 层约 86.11 MiB/request，43 个 backbone Block 约 901.29 MiB/request。这里是“Cache 也量化到 MXINT8”的条件估算，与仅把权重量化为 MXINT8 是两件事。

仍沿用“main Cache 为 FP8、Indexer-K 为 MXFP4、state 为 FP32”的 recipe，若完全照 TensorRT-LLM 的 role/page 设计，并要求每个 sliding state 至少占一个完整 page，page 对齐后的 dense-equivalent 上界为：

| `tokens_per_block` | 前 6 层 | 43 层 backbone |
|---:|---:|---:|
| 128 | 78.38 MiB | 820.19 MiB |
| 256 | 82.25 MiB | 859.13 MiB |

该表仍不等于 TensorRT-LLM 启动时立即为每个 request 实占的显存：TensorRT-LLM 从共享物理 page pool 按需分配。它更适合作为自研静态 dense buffer 或每 request 保底页设计的上界参考。

### 14.5 初始化值与“开辟不等于清零”

新 request-slot 初始化时：

```text
past_raw_len = 0
csa_count = 0
hca_count = 0
swa_valid_len = 0
all page-table entries / query indices = -1
Compressor KV state = 0
Compressor score state = -inf
```

CSA 的 previous-group A branch 初始 score 必须是 `-inf`，这样第一条 compressed entry 只使用 token 0～3 的 B branch。

65,536 条 CSA Cache 和 2,048 条 HCA Cache 的大数据区可以不清零：只要 count、page table、valid mask 和 request generation/epoch 保证未提交位置永远不会被读取。request-slot 复用时优先重置小 metadata 和 Compressor state，不要每次 memset 数百 MiB 的 full-history Cache。

静态图真正需要固定的是：

```text
持久 Cache 的最大地址范围
+ 当前 prefill256/decode1 的固定 workspace
+ 固定宽度 metadata/page table
```

不是把整个 256K 序列的所有 query、中间 Attention score 和 Top-K 结果同时常驻。若硬件允许固定 shape + 可变 page table，优先使用预分配共享 pool；只有物理地址也必须完全静态时，才按上述 `[layer,Bmax,capacity,dim]` 方式切出 monolithic 区间。

## 15. 自研静态图需要暴露哪些跨调用状态

若完全照语义实现，一个 `prefill256` / `decode1` 图的时间状态至少包括：

| 层类型 | 跨调用状态 |
|---|---|
| 全部层 | `past_raw_len`、每层 SWA ring/paged KV、SWA write position |
| CSA | main compressed cache/count、main Compressor KV+score state、Indexer-K cache/count、Indexer Compressor KV+score state |
| HCA | main compressed cache/count、main Compressor KV+score state |

推荐把同一请求所有 ratio 共用的相位直接由 `past_raw_len mod r` 推导，避免另外维护容易失配的 phase counter。

不要跨 Prefill/Decode 调用保存：

- mHC `post_mix/comb_mix`；它们只在当前 token/chunk 的层间传播；
- Prefill 每个 query 的 Top-K indices；它们用完即可释放；
- Attention 输出或 MoE 中间 activation。

### 15.1 静态图中最难的三个边界

真正需要专门 kernel/算子语义的不是整个 V4 Block，而是：

1. `compressor_update_finalize`：固定输入 256/1，masked 更新 paged state，只对完成窗口输出/scatter；
2. `indexer_score_topk`：读取 compressed Indexer-K，执行 causal score 和 Top-512；短历史可直接 dense fill；
3. `dual_pool_sparse_mla`：分别读取 SWA/Compressed pool，通过 indices + valid lengths + sink 做联合 online-softmax。

其余 mHC、RMSNorm、Linear、RoPE、MoE 都是固定 shape 下较常规的图节点，只是 mHC 的 Sinkhorn 和 MoE routing 需要对应高效 kernel。

### 15.2 “纯静态图”与 TensorRT-LLM 的差别

TensorRT-LLM 看上去支持 CUDA Graph，并不等于所有行为都由普通静态 Tensor 算子表达。它依赖：

- host/runtime 维护 request length 和 page table；
- graph-safe 的固定容量 workspace；
- mutable paged Cache；
- `-1` index 和 per-query valid length；
- Compressor、Top-K、索引转换、sparse MLA 等自定义 CUDA op；
- op 内部根据 mask/length 做状态更新。

所以正确的借鉴方式是固定外部 shape、把动态语义收进状态化算子，而不是强行将所有 Cache 拷贝、拼接和变长 Top-K 展开成普通静态图节点。

## 16. 前 6 个 Block 之外仍未覆盖的结构

前 6 个 Block 覆盖了主 decoder 中的三种 Attention、两种 router 和全部 mHC/MoE 基本结构，但不包括：

- Block 40～42 向 DSpark 提供的 target hidden states；
- checkpoint 中 `mtp.0/1/2` 的 DSpark draft stages；
- 最终 `HCHead`，它把 `[N,4,4096]` 收回 `[N,4096]`；
- final RMSNorm 和 LM head；
- speculative acceptance/rewind 对 Cache 的处理；
- 本 checkpoint 没有启用的 Engram 路径。

因此“前 6 层跑通”适合作为主干算子和 Cache 语义 bring-up，但不能等价于完整 0731/DSpark 端到端适配完成。

## 17. 建议的 bring-up 顺序

1. 先用 BF16/BF16 Cache 跑通 Block 0 的 mHC + SWA + Hash MoE；
2. 跑通 Block 0→1 的 deferred/fused mHC，验证与 unfused reference 对齐；
3. 单独验证 ratio4 main Compressor，重点覆盖 3/4、4/4、跨 chunk 的 1/4 状态；
4. 接入 CSA dual-pool Attention，先使用连续 compressed indices，不启用 Indexer；
5. 接入 Indexer Compressor、score、Top-K；用 `history<512 entries` 验证 dense fill 与 Top-K 等价；
6. 单独验证 ratio128 Compressor 的 127/128、128/128、129/128 边界；
7. 接入 HCA dual-pool Attention；
8. 跑 Block 0～5 Prefill-256，与官方 inference/TensorRT-LLM 中间 tensor 对齐；
9. 接 Decode-1，验证 position 256、259、383 三个关键 completion 边界；
10. 加 `valid_len`，覆盖 78、2359 和 43,456 三类 padded/chunked 输入；
11. 用 256K 容量初始化，验证最后一个合法 query 的 CSA count=65,536、HCA count=2,048、HCA index width=2,176，并检查越界保护；
12. 最后再切主 Cache 到 FP8、Indexer Cache 到目标格式、routed experts 到 MXINT8-B64。

建议逐层保存和比较：

```text
mHC layer_input
qr / raw_kv
main compressed output
Indexer compressed K / Top-K
Attention latent output
o_b output
MoE routed/shared/final output
mHC residual[4]
各 Cache count、position、state slot
```

## 18. 源码定位

- [DeepSeek-V4 TensorRT-LLM 使用说明](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/examples/models/core/deepseek_v4/README.md)
- [DeepSeek-V4 模型、权重映射、mHC/MoE forward](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/models/modeling_deepseekv4.py)
- [mHC Python 封装](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/modules/mhc/hyper_connection.py)
- [DeepSeek-V4 Attention 调度 hook](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/module.py)
- [Compressor](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/compressor.py)
- [V4 Indexer](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/indexer.py)
- [双池 Attention backend](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/backend.py)
- [本地索引到双池全局地址的转换](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/kernels.py)
- [DeepSeek-V4 metadata](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/metadata.py)
- [DeepSeek-V4 多角色 Cache Manager](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py)
- [NVIDIA DeepSeek-V4 技术博客](https://github.com/NVIDIA/TensorRT-LLM/blob/59e8079ab09423d4c700724fdd80ce45d8c76042/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md)
- [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)

## 19. xh2modelzoo 当前实现审计

> 审计日期：2026-08-11
> 审计仓库：`/data01/home/yujy/work/xh2modelzoo`
> 审计版本：`develop@72c2d6a1af4711c2d7ccc51f813aa8f3cb98cc0a`
> DeepSeek-V4 初始支持提交：`3d496d4c`；Pipeline Parallel 补充提交：`dfbab70b`

### 19.1 先给结论

xh2modelzoo 当前实现应定义为：

> 面向 `DeepSeek-V4-Flash-slim5l` 的固定长度、单次 Prefill + Decode bring-up 实现；mHC 数学和单次、对齐 Prefill 内的 CSA/HCA 压缩数学基本照搬 Hugging Face reference，但没有实现 CSA/HCA 跨调用所需的完整状态机。

它不是 TensorRT-LLM 那种完整的双池长上下文实现。当前代码通过以下简化避开了最难的静态状态更新：

1. Prefill 只生成一次固定长度的 main compressed KV；
2. Decode 只复用这份 compressed KV，不再运行 Compressor；
3. CSA 的 Indexer-K 只在 Prefill 图内临时存在，不保存为 Cache；
4. Decode 不运行 Indexer，也不重新做 Top-512；
5. main compressed KV 与普通 KV 在图中物理 `cat`，然后执行普通 dense Attention；
6. 没有 partial-window buffer、CSA previous-Ca overlap、compressed count 或有效长度。

因此，它能说明 mHC、CSA/HCA 的基础公式可以被 FX/HMONNX 展开，但不能说明固定 Prefill-256 的 chunk prefill、任意短输入 padding、Decode 持续更新或 256K 上下文已经解决。

当前能力边界如下：

| 场景 | 当前状态 | 判断 |
|---|---|---|
| mHC，Prefill/Decode 任意单次调用 | 数学结构完整，普通算子展开 | 基本可复用 |
| 恰好 256 个真实 token、`past=0` 的单次 Prefill | CSA 64 项、HCA 2 项，因果 mask 基本成立 | bring-up 可用 |
| 恰好 256 token 后的第一个 Decode token | compressed count 暂时不需增加，但 CSA/HCA 层 raw SWA 有实现偏差 | 仅部分正确 |
| Decode 连续越过 CSA/HCA completion 边界 | compressed KV 从不更新 | 不正确 |
| 78 token PAD 到 256 | Compressor 把 PAD 也压缩并保存 | 不正确 |
| 2359 / 43,456 token 多 chunk Prefill | 无跨 chunk Compressor 状态，兼容层切片也不完整 | 不支持 |
| 256K 上下文 | compressed 容量仍按 256 输入计算，Attention 还物理 dense concat | 不支持 |
| 完整 43 层 0731 主干 | HF Config 可识别 2 SWA + 21 CSA + 20 HCA | 结构可识别，未完整验证 |
| 0731 的 DSpark / `mtp.0/1/2` | converter 只导出主 CausalLM | 未支持 |

当前 README 明确以 5 层精简模型为对象：

```text
Layer 0,1: SWA
Layer 2,4: CSA
Layer 3:   HCA
context_length = 2048
prefill_length = 256
quant_type = w8a8h1_sefp
```

本地 0731 Config 经当前 Transformers 解析后，主干 43 层会得到：

```text
2  × sliding_attention
21 × compressed_sparse_attention
20 × heavily_compressed_attention
```

所以 converter 的按层遍历可以识别主干类型，但“能实例化 43 层”不等于 Cache 语义和长上下文已经正确。

### 19.2 代码入口与导出流程

真正的 XH2 实现不在 `examples_merak/llm/deepseekv4`。后者只是动态加载 checkpoint 自带 reference 并做前 6 层跟踪。XH2 HMONNX 实现位于：

```text
xh_model_zoo/xh_llm/models/deepseek_v4/
├── _hc.py                       # mHC / HyperHead
├── _model.py                    # Attention、RoPE、CSA/HCA 接入
├── _compressor_nl.py            # stateless CSA/HCA 非线性逻辑
├── _layers.py                   # Decoder / Model / CausalLM wrapper
├── _moe.py                      # Hash-MoE / learned MoE / shared expert
├── deepseek_v4_converter.py     # 两张固定图的导出
├── inference.py                 # Cache 创建与 Prefill/Decode 调度
└── deepseek_v4_hf_compatible.py # GenerationMixin 兼容层
```

导出过程为：

```text
HF checkpoint
  → AutoModelForCausalLM
  → DynamicModule wrapping
  → 拆出可量化 Linear / MoeBlock
  → 固定 P 的 Prefill 图
  → 固定 1 的 Decode 图
  → HMONNX
```

Prefill 和 Decode 是两份独立模型副本，分支在 FX trace 时由 `input_sequence_length == 1` 固化，不是运行时 `If`：

```text
Prefill graph: P = 256
Decode graph:  P = 1
```

converter 为每层分配一份普通共享 KV Cache：

```text
raw_kv[layer]: [1, 1, context_length, 512], FP16
```

再按 Prefill 图的固定长度计算 compressed shape：

```text
compressed_len[layer] = P // compress_rate[layer]

CSA: [1,1,64,512]
HCA: [1,1, 2,512]
```

这里使用的是 `P`，不是 `context_length`。即使把 `context_length` 配成 262,144，compressed Cache 仍然只有 64/2 项。

对于完整 43 层模型，Decode 调用的 Cache 参数布局是：

```text
past_key_caches =
    [raw_kv_0, ..., raw_kv_42,
     ckv_0,    ..., ckv_42]
```

即 43 份 raw Cache + 43 个 compressed 槽位。实际只有 41 个 compressor 层有有效 compressed tensor；两层 SWA 用 `[1,1,1,512]` 零张量占位。

这套实现中不存在以下持久 Cache：

```text
CSA Indexer-K Cache
CSA main partial-window KV/Gate state
CSA main previous-Ca overlap state
CSA Indexer partial-window / overlap state
HCA partial-window KV/Gate state
main / indexer compressed count
```

### 19.3 mHC 当前怎样计算

mHC 是当前实现里最完整、也最容易保留的一部分。

#### 19.3.1 输入展开

Embedding 输出先从：

```text
[B,S,4096]
```

复制成 4 路 residual stream：

```text
X = [B,S,4,4096]
```

整个 decoder 都保持这个 shape，最后才由 HyperHead 收回一条 stream。

#### 19.3.2 每个 mHC 的权重生成

每层有两套 mHC：Attention 前一套、MoE 前一套。对任意一套：

```text
flatten(X)                 [B,S,16384]
UnweightedRMSNorm(fp32)    [B,S,16384]
Linear                     [B,S,24]
split                      pre[4] + post[4] + comb[16]
```

其中：

```text
pre  = sigmoid(pre_logits  * scale[0] + base_pre) + eps
post = 2 * sigmoid(post_logits * scale[1] + base_post)
```

`comb_logits` reshape 为 `[B,S,4,4]`，然后：

1. 在最后一维做 softmax；
2. 做一次列归一化；
3. 再执行 19 轮行归一化 + 列归一化。

这与 checkpoint 的 `hc_sinkhorn_iters=20` 对齐，得到近似双随机的 `comb[B,S,4,4]`。

送入子层的单路输入是：

```text
collapsed[b,s,d] = Σ_h pre[b,s,h] * X[b,s,h,d]
```

#### 19.3.3 子层后的 4 路更新

Attention 或 MoE 输出 `Y[B,S,4096]` 后：

```text
X_new = post[...,h] * Y + comb^T @ X_old
```

shape 仍为 `[B,S,4,4096]`。一层的顺序是：

```text
attn_hc → input_layernorm → Attention → mHC post
ffn_hc  → post_attention_layernorm → MoE → mHC post
```

所有 decoder 完成后，HyperHead 再生成 4 个 collapse weights：

```text
[B,S,4,4096] → [B,S,4096] → final RMSNorm → LM head
```

#### 19.3.4 与 TensorRT-LLM 的差别

xh2modelzoo 是逐层、unfused 地执行完整 `pre → sublayer → post`。TensorRT-LLM 会把上一子层的 post 与下一子层的 pre 融合/延迟，以减少 4 路 residual 的访存。

两者数学目标相同，但当前 XH2 图会显式展开 Sinkhorn 的 19 轮归一化和多个 stream MatMul。mHC 没有时序 Cache，因此它不是 chunk prefill 的主要语义难点；后续可以先保留此 reference 图，再做融合算子优化。

### 19.4 CSA 当前怎样计算

以下以 `B=1, P=256, D=512, ratio=4` 为例。

#### 19.4.1 Main Compressor

Attention 输入 `H[1,256,4096]` 分别做：

```text
comp_kv_proj(H)    → KV   [1,256,1024]
comp_gate_proj(H)  → Gate [1,256,1024]
```

`1024 = 2 × 512`，两半分别表示 Ca 和 Cb。按 4-token group reshape：

```text
KV, Gate → [1,64,4,1024]
Gate += position_bias[4,1024]
```

第 `w` 个 compressed entry 组合：

```text
previous group 的 Ca: [1,4,512]
current  group 的 Cb: [1,4,512]
concat candidate axis: [1,8,512]
```

第 0 项没有 previous group，因此其 Ca value 塨 0、Ca gate 填 `-inf`。

softmax 的维度是 8 个 candidate 的轴，不是 512 个 channel 的轴：

```text
weight = softmax(Gate, dim=candidate_axis)
shape  = [1,64,8,512]
```

也就是 512 个 channel 各自拥有一组 8-way 权重。随后：

```text
Σ_candidate weight * value → [1,64,512]
RMSNorm(last_dim=512)
compress RoPE，anchor = 0,4,8,...,252
→ compressed_kv [1,1,64,512]
```

在同一次 256-token Prefill 内，当前代码能正确使用前一个 group 的 Ca；但这个 previous-Ca 只来自当前 tensor 的 `chunk[:, :-1]`，不会跨图调用保存。

#### 19.4.2 Lightning Indexer

Indexer 走另一套较小的 Compressor：

```text
idx_kv_proj(H)     [1,256,256] = 2 × 128
idx_gate_proj(H)   [1,256,256]
```

同样按 4 个 token 分组并做 Ca/Cb overlap：

```text
[1,64,4,256]
→ candidate [1,64,8,128]
→ channel-wise 8-way softmax
→ RMSNorm + compress RoPE
→ Indexer-K [1,64,128]
```

Query 与 head weight 为：

```text
idx_q_proj(q_residual) → Q [1,64 heads,256,128]
idx_w_proj(H)          → W [1,256,64]
```

当前实现计算完整 dense candidate score：

```text
dot        [1,64,256,64]
ReLU * 1/sqrt(128)
* W / sqrt(64)
reduce(head=64)
→ index_score [1,256,64]
```

之后做 causal mask 与 Top-K，再通过 OneHot + ReduceSum 生成：

```text
block_bias [1,1,256,64]
```

需要注意，固定 256 时 candidate 总数只有 64，而 `index_topk=512`：

```text
top_k = min(512, 64) = 64
```

因此首次 Prefill 的 Top-K 实际上不会产生稀疏收益，只是在因果约束下选择全部已完成 entry。真正需要从大量历史中选 Top-512 的场景没有被当前图覆盖。

#### 19.4.3 送入主 Attention

xh2modelzoo 把 main compressed KV 物理拼到普通 KV 前面：

```text
KV layout = [compressed_kv | raw_kv | zero_padding]
```

然后把 single-head KV expand 到 64 个 Q head，执行普通：

```text
QK^T → add mask → SinksMaskedSoftmax → PV
```

Prefill 的 mask 是：

```text
[CSA block_bias | raw causal-SWA128 mask]
```

最后对 Attention output 的后 64 维执行 conjugate RoPE，再走 8 组 `o_a` 与一个 `o_b`。

这不是双池 sparse MLA kernel；compressed/raw 只是两段 dense tensor。对默认 2K context 尚可用于 bring-up，但静态 context 扩到 256K 时，仅 `[64 heads,256 queries,约262K keys]` 的 logits 就超过 42 亿元素，无法作为生产实现。

### 19.5 HCA 当前怎样计算

HCA 没有 Ca/Cb overlap，也没有 Indexer。固定 256 时：

```text
comp_kv_proj(H)    → KV   [1,256,512]
comp_gate_proj(H)  → Gate [1,256,512]

reshape            → [1,2,128,512]
Gate += position_bias[128,512]
softmax(dim=128-token candidate axis)
channel-wise weighted sum
RMSNorm
compress RoPE，anchor = 0,128
→ compressed_kv [1,1,2,512]
```

HCA 的 `block_bias[1,1,256,2]` 在图构建时预计算：query `t` 只允许看到满足以下条件的 entry `w`：

```text
w < floor((t + 1) / 128)
```

例如：

```text
t=126: 0 项
t=127: entry 0
t=255: entry 0,1
```

单次 256-token、起始位置为 0 时，这段数学是成立的。问题同样在跨调用：代码不保留不足 128 个 token 的 KV/Gate，也不把新完成的 HCA entry append 到历史 Cache。

### 19.6 Prefill-256 与 Decode-1 的实际 Cache 流

#### 19.6.1 单次、恰好 256 个真实 token

Prefill 图执行后：

```text
每层 raw KV Cache 写入 position 0..255
每个 CSA 层输出 64 个 main compressed KV
每个 HCA 层输出 2 个 main compressed KV
```

Inference engine 把 Prefill 的额外输出写入 Python 侧 `compressed_kv_caches[layer]`。

切到 Decode 图后，调用参数变成：

```text
[43 raw caches] + [43 compressed slots]
```

Decode 图内部仅执行：

```text
cached compressed KV + updated raw KV → cat → Attention
```

Decode 图只输出 logits，没有新 compressed KV 输出。因此 Python runtime 也没有机会更新 compressed Cache。

#### 19.6.2 三个关键位置

以初始 prompt 恰好 256 token 为例：

| Query position | 正确语义 | 当前 xh2modelzoo |
|---:|---|---|
| 256 | CSA 仍 64 项；HCA 仍 2 项 | compressed count 暂时一致 |
| 259 | 收齐 256～259，CSA 应新增第 65 项 | 仍为 64 项 |
| 383 | 收齐 256～383，HCA 应新增第 3 项 | 仍为 2 项 |

所以“Decode-1 在 position 256”没有立即暴露 compressed 更新缺失；连续 Decode 必须至少覆盖 259 和 383 才能验证完整状态机。

#### 19.6.3 CSA/HCA 层的 raw SWA 还有独立偏差

所有 V4 Attention 类型都应包含最近 128 个 raw KV。当前 `_DeepseekV4Attention._setup()` 却只对纯 `sliding_attention` 层设置：

```python
attention_max_length = 128
```

CSA/HCA 层被设置成 `-1`，因此 Decode 会读取完整 raw history，再额外读取 compressed history，而不是：

```text
最近 128 raw + compressed history
```

单次 Prefill 依靠显式 mask 仍限制了 raw SWA128；Decode 没有显式 mask，因此这个差异在历史长度超过 128 后会直接出现。

### 19.7 Padding 和多 chunk 为什么不正确

#### 19.7.1 78 token PAD 到 256

`prepare_inputs()` 会用 EOS PAD 到 256，并正确传入：

```text
current_input_length = 78
```

普通 `LLMCacheV2` 会使用这个长度控制 raw Cache 写入，最后 logits 也 gather position 77。但是 CSA/HCA Compressor 虽然接收了 `current_input_length` 参数，非线性实现并没有使用它，而是直接使用静态：

```text
seq_len = self.input_sequence_length = 256
```

因此当前结果是：

```text
CSA 固定输出 64 项，而正确值是 floor(78/4)=19 项，remainder=2
HCA 固定输出  2 项，而正确值是 floor(78/128)=0 项，remainder=78
```

其中 CSA entry 19 已混入 PAD 76～79 中的后两个位置；HCA entry 0 也混入了 78～127 的 PAD。Prefill 的 query 77 通过 block bias 看不到这些未来项，所以 Prefill last-token 可能仍较接近 reference；进入 Decode 后却把固定 64/2 项全部视为有效，PAD 污染会暴露出来。

#### 19.7.2 2359 或 43,456 token

HF compatible wrapper 表面上把长输入 pad 成 256 的倍数并循环调用 Prefill 图，但当前实现存在多层问题：

1. 每个 sub-call 只切了 `inputs_embeds`，没有同步切 `position_ids` 和 `input_ids`；
2. Prefill static mask 按 chunk 内 `0..255` 预构建，没有按 `past_seq_length` 平移；
3. Compressor RoPE anchor 每次都从 `0,4,...` 或 `0,128` 重新开始；
4. 每次 Prefill 输出会覆盖上一份 compressed Cache，不会 append；
5. CSA 没有保存上一 chunk 最后一个 group 的 Ca；
6. CSA/HCA 都没有保存最后不足 4/128 的 remainder；
7. CSA Indexer-K 也不会跨 chunk 保存和增长。

所以 2359 和 43,456 token 不是“效率较差但还能跑”，而是当前状态模型本身不成立。以 43,456 为例，正确结果应为：

```text
CSA count = 10,864
HCA count =    339，另有 64-token open group
```

当前模型的 compressed 输入 shape 仍固定为 64/2，无法表示这两个 count。

### 19.8 当前验证覆盖到哪里

仓库中没有 DeepSeek-V4 的常规单元测试或 CI 用例，只有 examples 下的手工验证脚本。

README 记录的验证对象是 5 层 slim 模型，主要指标为：

```text
wrapped PT vs HF: >0.999
ONNX Prefill:     约0.9995
ONNX Decode 1..8: 约0.97～0.99
```

需要注意：

- `verify_wrapped_pt.py` 验证的是单次 stateless forward，正好是当前实现最接近 HF fallback 的路径；
- `verify_prefill_decode_cosine.py` 要求 prompt 小于 256，只跑默认 8 个 decode step；
- 脚本只打印 cosine，没有 `assert` 失败门槛；
- 没有测试 78、256、259、383、2359、43,456 或 256K；
- 没有测试完整 43 层和 DSpark/MTP。

因此 README 中的 cosine 可以作为基础算子 bring-up 证据，不能作为 chunked long-context 正确性的证据。

### 19.9 哪些代码可保留，哪些需要重做

#### 可以保留并继续验证

1. mHC / HyperHead 的 shape 和基础公式；
2. main/compress 双 RoPE cache 与 interleaved RoPE 分解；
3. Q、shared-KV、conjugate RoPE、grouped output projection；
4. CSA/HCA 线性投影拆分及单窗口 softmax-gating reference；
5. Hash-MoE、learned Top-K MoE、shared expert 的基本 wrapper；
6. Prefill/Decode 分图与 `CacheTensor` 接口框架。

#### 必须替换或扩展

1. 把 `_compressor_nl.py` 的 stateless 接口改为带 persistent state 的更新接口；
2. 为所有层建立真正的 SWA128 ring/paged raw Cache；
3. CSA 增加 main compressed Cache、Indexer-K Cache、两套 partial/overlap state 和 count；
4. HCA 增加 main compressed Cache、partial state 和 count；
5. Prefill/Decode 都必须调用同一套 Compressor update 语义；
6. 用 `valid_len` 隔离 PAD，只有完成窗口才能 commit compressed entry；
7. 避免按 256K key 轴做 dense physical concat，改成双池 sparse Attention，或两个分支的 online-softmax 合并；
8. 增加跨 completion 边界和多 chunk 的逐层 golden test。

最准确的工程判断是：

> 当前实现可以作为 mHC 和单窗口 Compressor 的算子级 reference，但不应直接在其固定 64/2 compressed tensor 设计上继续堆补丁来做 256K。长上下文部分需要重新定义 Cache ABI 和状态更新语义。

### 19.10 xh2modelzoo 源码定位

- mHC 计算：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_hc.py:51-83`
- Decoder 中两套 mHC：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_layers.py:75-102`
- Model 的固定 Prefill/Decode 分支：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_layers.py:131-257`
- CSA/HCA 与普通 KV 的物理拼接：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_model.py:220-311`
- CSA Compressor 与 Indexer：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_compressor_nl.py:149-448`
- HCA Compressor：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_compressor_nl.py:456-565`
- raw SWA Cache 配置：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/_model.py:420-440`
- 固定 compressed shape 的导出：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/deepseek_v4_converter.py:642-680,761-838`
- Runtime 保存/回灌 compressed KV：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/inference.py:99-108,262-291`
- 长输入兼容层循环：`/data01/home/yujy/work/xh2modelzoo/xh_model_zoo/xh_llm/models/deepseek_v4/deepseek_v4_hf_compatible.py:100-154`
- 当前示例与验证边界：`/data01/home/yujy/work/xh2modelzoo/examples/llm/deepseek_v4/README.md`

## 二十、从零重做：Prefill-256 / Decode-1 的静态图 Cache ABI

这一节不再延续第十九节审计出的旧实现，而是重新定义一套可用于自研芯片的静态图方案。固定条件为：

```text
Prefill 图 P = 256
Decode 图  D = 1
SWA window W = 128
CSA 压缩率 R4 = 4，Top-K = 512
HCA 压缩率 R128 = 128
最大上下文 Lmax 在 wrap 时给定
```

先给出结论：

1. mHC 的 Sinkhorn 不应改算法，也不建议用纯 FP16 跑 20 轮；正确的快法是把 4×4 Sinkhorn 保持 FP32、完全展开并与前后小算子融合。
2. SWA 在一个 Prefill chunk 内真正需要的 KV 并不是 `P + P`，而是 `W - 1 + P = 383`；如果芯片的 K 轴要求 256 对齐，物理宽度才取 512。
3. 用户提出的 CSA `8-token state + offset` 思路成立；`8 + 256` 对齐后固定取 260，恰好形成 65 个 4-token group，并产生最多 64 个压缩项。
4. `-65504` 虽不是 IEEE FP16 的 `-inf`，但在本方案的 `offset + num_new/candidate_valid` 约束下可以安全用作 Compressor Gate 的无效填充值；无需再增加逐行 mask。
5. CSA Main Compressor 和 CSA Indexer Compressor 必须各有一套 8-token state；它们的投影宽度和长期 Cache 都不同。
6. CSA 的 Top-512 对每个 query 都可能不同，因此 Prefill 时不能把选出的 512 项当作一份所有 query 共享的普通 KV Tensor。
7. SWA 与 compressed 分支必须进行一次联合 Softmax，并共同加入 attention sink；不能分别算两个 Attention 后直接相加。

### 20.1 统一的长度、phase 和写入公式

定义：

```text
past_len  = 本次调用前已经处理的真实 token 数
valid_len = 本次固定 P/D 输入中真实 token 数，不含 PAD
r         = 压缩率，CSA 为 4，HCA 为 128
```

所有 Compressor 都使用同一组整数公式：

```text
phase_r       = past_len % r
offset_r      = r - phase_r                 # 取值 1..r，phase=0 时 offset=r
num_new_r     = (phase_r + valid_len) // r
write_start_r = past_len // r
write_end_r   = write_start_r + num_new_r
next_phase_r  = (phase_r + valid_len) % r
next_offset_r = r - next_phase_r
```

其中 `num_new_r` 也可写成：

```text
num_new_r = floor((past_len + valid_len) / r) - floor(past_len / r)
```

这比用多套 Prefill/Decode 特判更可靠。用户给出的 offset 递推：

```text
next_offset = r - (valid_len + (r - old_offset)) % r
```

与上式等价。

对于最大上下文：

```text
CSA Main Cache capacity    N4   = floor(Lmax / 4)
CSA Indexer-K capacity     N4   = floor(Lmax / 4)
HCA Main Cache capacity    N128 = floor(Lmax / 128)
```

如果底层 Cache 以 256 个原始 token 为一个逻辑页，可再将：

```text
N4   向上对齐到 64 entries/page
N128 向上对齐到 2 entries/page
```

### 20.2 第一刀：mHC Sinkhorn

#### 20.2.1 它实际只处理一个 4×4 矩阵

对每个 token，mHC residual 为：

```text
X: [4, 4096]
flatten(X): [16384]
```

经过 RMSNorm 和一个 `16384 -> 24` 的线性投影后，24 个值分成：

```text
pre logits:  [4]
post logits: [4]
comb logits: [4, 4]
```

其中：

```text
pre  = sigmoid(pre_logits) + eps
post = 2 * sigmoid(post_logits)
comb = Sinkhorn20(comb_logits)
```

真正的大头是 `16384 x 24` 的生成矩阵乘；Sinkhorn 自身每个 token 只有 16 个数。即使做 20 轮行列归一化，其算术量也远小于前面的线性投影。

#### 20.2.2 GitHub 上的“快算法”是什么

目前可直接参考的高性能实现有：

- [TileLang DeepSeek mHC example](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc/example_mhc_pre.py)
- [vLLM TileLang mHC kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/mhc/tilelang_kernels.py)
- [SGLang mHC implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/mhc.py)

它们并没有用 Greenkhorn、Newton 或减少迭代次数来替换模型定义，而是：

1. 前置 RMS、24 维 projection accumulation 使用 FP32；
2. 4×4 stable softmax 使用减 row-max；
3. 20 轮行列归一化完全展开在寄存器中；
4. `pre/post/comb` 的生成和 residual mixing 尽量融合；
5. 只在最终 layer input / residual 输出处转回 BF16。

这才是这里合适的“快速 Sinkhorn”。对于一个固定的 4×4 问题，换收敛算法节省不了多少总时延，反而会改变 checkpoint 的函数。

#### 20.2.3 纯 FP16 为什么有风险

官方 inference kernel 和当前 vLLM 实现均把以下量声明为 FP32：

```text
hc_fn / hc_scale / hc_base
RMS accumulation
24 个动态 logits
4×4 Sinkhorn 的 exp、sum、divide 和中间矩阵
post_mix / comb_mix
```

本地 checkpoint 的 mHC 参数也说明不能只看“4×4 很小”就直接降精度：

```text
92 个 mHC site（46 组 block 参数，含主干外附加 block）
comb base min = -45.4563
comb base max =  13.5131
```

在 FP16 中，相对最大值小约 17 以上的指数项就可能进入 subnormal 或下溢；若芯片开启 FTZ，subnormal 会直接变成 0。模型中的部分 comb logits 差值远大于这个范围。

使用 checkpoint 的 `hc_fn/hc_scale/hc_base` 构造协方差，并采样 5,888 个归一化 residual 的 4×4 logits，做了一个非端到端风险筛查：

| Sinkhorn 实现 | 相对 FP32 reference 的 max abs error | P99 abs error |
|---|---:|---:|
| 全部 FP16，不模拟 FTZ | 1.59e-3 | 1.07e-3 |
| 全部 FP16，模拟 subnormal FTZ | 1.58e-1 | 3.04e-3 |
| logits 先舍入 FP16，Sinkhorn 转 FP32 | 3.36e-4 | 8.74e-5 |

这只是用真实权重统计量构造的数值压力测试，不等价于最终模型精度；但它已经足够说明：

> 纯 FP16 Sinkhorn 的普通样本可能看似稳定，但 FTZ 下存在明显离群风险，不应直接作为默认实现。

#### 20.2.4 推荐精度契约

第一版建议：

```text
residual input/output:             FP16 或 BF16
16384 x 24 GEMM input/weight:      可评估 FP16/BF16
GEMM accumulation/output logits:   FP32
stable softmax + Sinkhorn20:       FP32
pre/post/comb residual accumulate: FP32
最终 layer input/residual:         cast 回模型激活 dtype
```

如果芯片只能高效做 FP16 GEMM，可以先让大 GEMM 用 FP16 输入、FP32 accumulation，而不要为了统一 dtype 把只有 16 个元素的 Sinkhorn 也降成 FP16。

验收至少检查：

```text
所有中间值 finite
comb 与官方 FP32 reference 的 max/cosine
每层 mHC 输出 cosine
43 层最终 logits cosine / token agreement
Prefill-256 与 Decode-1 两种 token 数
FTZ on/off
```

### 20.3 第二刀：SWA128 不需要把“语义长度”改成 512

#### 20.3.1 一个 Prefill chunk 的最小 KV union

窗口大小 128 表示 query `t` 可见：

```text
[t - 127, ..., t]
```

一个 256-token chunk 的第一个 query 需要前 127 个历史 token，最后一个 query 需要当前 chunk 的全部前缀。因此整个 chunk 的 KV union 是：

```text
(W - 1) + P = 127 + 256 = 383
```

不是 128，也不必天然是 512。

xhquant 当前 Hybrid Cache 的核心公式正是：

```text
aligned(attention_max_length + input_seq_len - 1, 16)
```

即 `align_up(383, 16) = 384`。它在更新时保留最后 `W - 1` 个历史项，再追加固定 shape 的 current states。

但当前实现不适合直接复用到新静态 ABI，原因是：

1. `HybridCacheTensor.cache_valid_len` 是 Python 成员；
2. 路径中有 `int(current_seq_length.item())`；
3. capture-safe 分支明确标记 HybridCache 尚未实现；
4. 它把 persistent tail、当前 chunk 和 Attention view 混在同一个 buffer 中。

#### 20.3.2 推荐把 SWA 拆成 tail state 和临时 Attention pool

每层只持久化：

```text
swa_tail: [B, 127, 512]
```

有效历史 token 在 tail 中右对齐，初始全部无效。Prefill 中：

```text
raw_cat  = concat(swa_tail, current_raw_kv)   # [B, 383, 512]
raw_pool = pad(raw_cat, K_swa_prefill)        # [B, 512, 512]，若 K 轴按 256 对齐
```

所以用户提出的 Prefill raw 宽度 512 可以保留，但应定义为：

```text
K_swa_prefill = align_up(W - 1 + P, hardware_k_align)
```

当 `hardware_k_align=256` 时结果为 512；这是硬件对齐，不是因为 `window < chunk` 导致 384 “放不下”。

Decode 中：

```text
raw_pool = concat(swa_tail, current_raw_kv)   # [B, 128, 512]
```

更新 tail 的公式非常简单，且输出 shape 永远固定：

```text
swa_tail_next = raw_cat[:, valid_len : valid_len + 127]
```

例子：

- `valid_len=256`：丢掉旧 tail 和 current 的前 129 个位置，留下 current 的最后 127 项；
- `valid_len=78`：留下旧 tail 的最后 49 项，再接 current 的 78 个真实项；
- Decode `valid_len=1`：丢掉最老 1 项，追加当前 token。

#### 20.3.3 SWA mask

固定 tail slot `j in [0,126]` 的绝对位置为：

```text
k_abs(j) = past_len - 127 + j
```

current slot `127+u` 的绝对位置为：

```text
k_abs(127+u) = past_len + u
```

query `q` 的绝对位置为：

```text
t = past_len + q
```

SWA mask 的有效条件为：

```text
q < valid_len
max(0, t - 127) <= k_abs <= t
current slot 还需 u < valid_len
```

因此 Prefill 即使把整个 256-token current KV 一次性放入 raw pool，也不会看到未来 token。

### 20.4 第三刀 A：CSA Main Compressor 的静态更新

#### 20.4.1 8-token state 到底存什么

Main Compressor 的 head dimension 为 `d=512`，因为 overlap，其投影为：

```text
U = Wkv(H):   [B, P, 2d] = [B, P, 1024]
Z = Wgate(H): [B, P, 2d] = [B, P, 1024]
Z += APE[absolute_position % 4]
```

持久状态应为：

```text
main_u_state: [B, 8, 1024]
main_z_state: [B, 8, 1024]
```

也可以物理 pack 为 `[B,8,2048]`，但逻辑上仍要区分 KV 值和 Gate logits。

状态 invariant：

```text
保存最近 8 个真实 token 的 raw compressor projection；
不足 8 个时有效项右对齐；
无效 U = 0；
无效 Z = -65504；
只有 candidate_valid 的压缩结果允许写入 Cache。
```

在当前固定 offset 方案中，`-65504` 可以直接用作 Gate 的无效填充值：

1. 第一次 Prefill 虽然输入 state 的 8 行全无效，但 `offset_4=4`，`G0` 只取 state 的后 4 行；第一个实际 candidate 是 `G0` 的 4 行无效分支加 `G1` 的 4 行当前有效分支，不存在一个需要提交的 8 行全无效 candidate；
2. 短尾 chunk 后部确实可能静态算出 8 行全为 `-65504` 的 candidate，但它满足 `i >= num_new_4`，不会写入 Cache，也不会进入后续 Attention；
3. Decode 非压缩边界产生的 candidate 同样由 `num_new_4=0` 丢弃。

因此这里不需要为了 all-invalid softmax 再增加逐行 boolean mask，`num_new_4/candidate_valid` 已经构成最终保护。仍需区分概念：`-65504` 是 FP16 最小有限值，不是 IEEE `-inf`；若未来移除 offset/候选写入约束，才需要重新审视 all-invalid 行为。

实现时只需保证：

```text
无效 branch 的 Gate = -65504
candidate_valid = arange(64) < num_new_4
CacheUpdate 只提交 candidate_valid=True 的项
```

#### 20.4.2 checkpoint 的精确 overlap 公式

把对齐后的 raw projection 按 4 token 分组：

```text
G_i^U: [B, 4, 1024]
G_i^Z: [B, 4, 1024]
```

当前 checkpoint/inference 代码对第 `i` 个输出实际拼接：

```text
V_i = concat(G_i^U[..., :512], G_{i+1}^U[..., 512:])  # [B, 8, 512]
L_i = concat(G_i^Z[..., :512], G_{i+1}^Z[..., 512:])  # [B, 8, 512]
W_i = softmax(L_i, dim=8-token-axis)                   # [B, 8, 512]
C_i = sum(W_i * V_i, dim=8-token-axis)                 # [B, 512]
```

即：每一个输出 feature 都对 8 个 token/branch 候选单独做 Softmax；不是对 512 个 feature 做 Softmax，也不是 8 个 token 共用一个 scalar weight。

技术报告把两组 projection 记为当前组的 `a` 分支和上一组的 `b` 分支；checkpoint 拼接顺序由权重转换决定，实现时应按上述本地 inference 代码的 half layout，而不要只凭变量名猜 `a/b`。

#### 20.4.3 为什么 `8 + 256 -> slice 260` 正好成立

Prefill：

```text
u_cat = concat(main_u_state, current_u)  # [B, 264, 1024]
z_cat = concat(main_z_state, current_z)  # [B, 264, 1024]
```

按 `offset_4` 固定取 260 行：

```text
u_aligned = Gather(u_cat, offset_4 + arange(260))
z_aligned = Gather(z_cat, offset_4 + arange(260))
```

四种 phase 的意义如下：

| `past_len % 4` | `offset_4` | 对齐起点 |
|---:|---:|---|
| 0 | 4 | 上一个完整 group 的开头 |
| 1 | 3 | 上一个完整 group 的开头 |
| 2 | 2 | 上一个完整 group 的开头 |
| 3 | 1 | 上一个完整 group 的开头 |

260 行 reshape 为：

```text
[B, 65 groups, 4, 1024]
```

相邻 group 两两组合：

```text
(G0,G1), (G1,G2), ..., (G63,G64)
```

因此静态地产生 64 个 candidate：

```text
candidate_main: [B, 64, 512]
```

高效形成 8-token 轴不需要通用的滑窗展开，但必须先明确两组 512-d branch。若 projection/state 使用 packed layout：

```text
u_aligned_packed: [B,260,1024]
z_aligned_packed: [B,260,1024]
```

先 Split 最后一维：

```text
u_prev_all = u_aligned_packed[...,   0:512]  # [B,260,512]
u_curr_all = u_aligned_packed[..., 512:1024] # [B,260,512]
z_prev_all = z_aligned_packed[...,   0:512]
z_curr_all = z_aligned_packed[..., 512:1024]
```

若线性投影一开始就输出两份独立 Tensor，则直接从上述四个 `[B,260,512]` Tensor 开始。接着在 token 轴 Slice 两次：

```text
u_prev = u_prev_all[:,   0:256, :].reshape(B,64,4,512)  # G0..G63
u_curr = u_curr_all[:,   4:260, :].reshape(B,64,4,512)  # G1..G64
z_prev = z_prev_all[:,   0:256, :].reshape(B,64,4,512)
z_curr = z_curr_all[:,   4:260, :].reshape(B,64,4,512)

u8 = concat(u_prev, u_curr, axis=2)   # [B,64,8,512]
z8 = concat(z_prev, z_curr, axis=2)   # [B,64,8,512]
w8 = softmax(z8, axis=2)
candidate_main = reduce_sum(w8 * u8, axis=2)
```

也可以按“先 reshape、再在 group 轴 Slice”的方式写，二者只是 view 的顺序不同：

```text
u_prev_groups = u_prev_all.reshape(B,65,4,512)
u_curr_groups = u_curr_all.reshape(B,65,4,512)
z_prev_groups = z_prev_all.reshape(B,65,4,512)
z_curr_groups = z_curr_all.reshape(B,65,4,512)

u8 = concat(
    u_prev_groups[:, 0:64, :, :],  # G0..G63 的 previous branch
    u_curr_groups[:, 1:65, :, :],  # G1..G64 的 current branch
    axis=2,
)                                      # [B,64,8,512]
z8 = concat(
    z_prev_groups[:, 0:64, :, :],
    z_curr_groups[:, 1:65, :, :],
    axis=2,
)                                      # [B,64,8,512]
```

注意这里是对两份 branch Tensor 各取一次 Slice；不能对同一个 `[B,65,4,512]` 的 token/group 轴取两段，就把它们当成两组投影。相邻关系与 branch 关系是两个独立维度。

所以如果变量 `u_aligned` 本身只有 `[B,260,512]`，它只能表示其中一个 branch；还必须有另一份 `[B,260,512]`。一个 512-d Tensor 不能直接 reshape 出 `2×512`。

这里四个 token-axis Slice 都是规则 view，只有 4+4 的 Concat 需要形成新布局。单个 `[B,64,8,512]` 在 B=1、FP16 下只有 512 KiB，correctness 版本直接这样做是合理的。

如果要进一步避免物化 `u8/z8`，可把 8-way Softmax 拆成两个 4-way branch reduction，结果完全等价：

```text
m = maximum(reduce_max(z_prev, axis=2),
            reduce_max(z_curr, axis=2))            # [B,64,1,512]

e_prev = exp(z_prev - m)
e_curr = exp(z_curr - m)
denom = reduce_sum(e_prev, axis=2) + reduce_sum(e_curr, axis=2)

candidate_main = (
    reduce_sum(e_prev * u_prev, axis=2)
  + reduce_sum(e_curr * u_curr, axis=2)
) / denom
```

这一版只使用 Slice、ReduceMax、Exp、ReduceSum 和逐元素算子，适合编译器融合成 Compressor kernel。Indexer 只需把 512 换成 128，布局完全相同。

只有前 `num_new_4` 个 candidate 有效：

```text
candidate_valid[i] = i < num_new_4
```

这证明用户提出的 `slice 260` 是正确的。无需动态改变输出 shape。

#### 20.4.4 下一次调用的 8-token state

下一状态也不需要复杂分支：

```text
main_u_state_next = u_cat[:, valid_len : valid_len + 8]
main_z_state_next = z_cat[:, valid_len : valid_len + 8]
```

例如旧 state 保存 token 70..77，下一次 `valid_len=256` 时，slice 会留下本次 current 的最后 8 个真实 token；Decode `valid_len=1` 时则留下旧 state 的后 7 项和当前 1 项。

#### 20.4.5 Decode-1 的正确写入条件

Decode 图的 cat 只有 `8 + 1 = 9` 行。为了固定取 8 行，可在右侧追加 3 个 invalid row：

```text
decode_cat_pad: [B, 12, 1024]
aligned8 = Gather(decode_cat_pad, offset_4 + arange(8))
candidate_main: [B, 1, 512]
```

写入数为：

```text
num_new_4 = (phase_4 + 1) // 4
```

所以：

```text
旧 phase=0/1/2 -> 不写
旧 phase=3     -> 写 1 项
```

如果用 offset 表述：

```text
旧 offset=1，或者更新后的 next_offset=4 时写 1 项
```

不是“旧 offset=4 时写”。

#### 20.4.6 Main Cache 定点写入与 RoPE

压缩结果先做：

```text
RMSNorm(512)
last 64 dims partial RoPE
KV Cache quant/store
```

第 `j` 个 compressed entry 的 RoPE anchor 是：

```text
compressed_pos = 4 * j
```

本次 candidate `i` 的全局 entry index 为：

```text
j = write_start_4 + i
```

只写：

```text
i < num_new_4
```

长期 Cache：

```text
csa_main_cache: [B, N4, 512]  # 物理上可为 FP8 NoPE + BF16 RoPE 的混合布局
```

### 20.5 第三刀 B：CSA Lightning Indexer

#### 20.5.1 它不是第二个 Attention

Indexer 是一个学习得到的“检索器”。它只回答：

> 对当前 query，历史上哪 512 个 C4 compressed block 最值得送入真正的 512-d Attention？

它本身不会产生最终 attention value output。

CSA 每个 C4 block 因此有两份不同的长期表示：

```text
Main compressed KV: [512]  # 最终 Attention 使用
Indexer compressed K: [128] # 仅用于算检索分数
```

两份 Cache 的 entry index 一一对应。Indexer 选出全局 index `j` 后，用同一个 `j` 去 Main Cache gather 512-d KV。

#### 20.5.2 Indexer Compressor 也需要独立的 8-token state

Indexer Compressor 与 Main 使用相同的 C4 overlap 时序，但 projection 宽度不同：

```text
index_u = Wkv_index(H):   [B, P, 256]  # 2 * 128
index_z = Wgate_index(H): [B, P, 256]

index_u_state: [B, 8, 256]
index_z_state: [B, 8, 256]
```

它不能复用 Main 的 `[8,1024]` state。它同样使用：

```text
offset_4
Prefill 固定 slice 260 -> 64 candidates
Decode 固定 slice 8   -> 1 candidate
同一 write_start_4 / num_new_4
```

压缩后的 128-d Indexer K 执行：

```text
RMSNorm
last 64 dims compressed RoPE，anchor=4*j
Hadamard rotation
FP4 QAT/量化路径
写 indexer_k_cache[j]
```

长期 Cache：

```text
indexer_k_cache: [B, N4, 128]
```

#### 20.5.3 每个 query 如何打分

对当前 raw query token `t`：

```text
qr_t:                [1024]      # 与主 Attention 共享的 Q latent
Q_index = Wq_index(qr_t)
Q_index:             [64, 128]
head_weight = Ww(H_t)
head_weight:         [64]
```

Q 执行 raw position `t` 的 partial RoPE、Hadamard rotation 和 FP4 QAT/量化。

官方 inference 中把 scale 合入 head weight：

```text
head_weight *= 1 / sqrt(128) / sqrt(64)
```

对每个历史 compressed block `j`：

```text
dot[t,h,j]   = dot(Q_index[t,h,:], indexer_k_cache[j,:])
relu[t,h,j]  = max(dot[t,h,j], 0)
score[t,j]   = sum_h(head_weight[t,h] * relu[t,h,j])
```

shape 为：

```text
Q_index: [B, P, 64, 128]
K_index: [B, N4, 128]
score:   [B, P, N4]
```

这里有三个容易误解的点：

1. ReLU 在每个 head 的 dot product 后、head reduction 前；
2. 64 个 head 最后汇成每个 compressed block 的一个 scalar score；
3. block 之间不做 Softmax，直接对 scalar score 做 Top-K。

#### 20.5.4 Chunk Prefill 的因果长度

虽然本次最多 64 个新 Indexer K 可以先统一写入 Cache，但 query `t` 只能检索已经完成的 C4 block：

```text
n4(t) = floor((t + 1) / 4)
valid compressed index j: 0 <= j < n4(t)
```

其中 `t` 为 0-based absolute token position。

例如第一次 256-token Prefill：

```text
query 0..2:   n4=0
query 3..6:   n4=1
...
query 255:    n4=64
```

所以 TopK 前要对每个 query 分别 mask：

```text
compressed_valid[t,j] = j < n4(t)
masked_score = where(compressed_valid, score, MASK_MIN)
```

第一版可以直接对 `[B,P,N4]` 使用普通 `TopK(k=512)`。TopK 后再生成：

```text
topk_idx_raw    = selected_index                  # 始终是 0..N4-1
topk_valid      = topk_idx_raw < n4(t)
topk_gather_idx = where(topk_valid, topk_idx_raw, 0)
```

当历史少于 512 项时，真实有效个数就是 `min(n4(t),512)`。无效 slot 使用安全 index 0 完成 Gather，最终由 `topk_valid` 屏蔽；不要把 `-1` 直接送给不支持负索引的 Gather。若外部 ABI 希望用 `-1` 表示无效，可以在 Gather 之后再导出该形式。

`MASK_MIN` 必须在 TopK 前小于所有有效 score，否则无效位置可能提前挤掉一个有效位置，TopK 后再 mask 已无法找回。可选精度契约为：

```text
方案 A：score/TopK 支持 infinity，MASK_MIN = -inf
方案 B：score 为 FP16，并保证有效 score > -65504，MASK_MIN = -65504
```

方案 B 对当前模型是可行方向，但“保证”应成为算子契约而不是口头假设。建议在校准集记录 Indexer score 的 min/max；如需形式化保证，可在不触及观测分布的远端阈值做 Clip，例如有效 score 限制在 `[-32752,32752]`，再用 `-65504` mask。这样无效值严格小于任一有效值。

#### 20.5.5 静态图 TopK 的真正难点

在 256 Ki context 下：

```text
N4 = 65,536
```

如果直接物化最终 head-reduced score：

```text
[1,256,65536]
FP16 = 32 MiB
FP32 = 64 MiB
```

这在 B=1 时并非完全不可接受。真正不能物化的是：

```text
[1,256,64,65536]
FP16 = 2 GiB
FP32 = 4 GiB
```

因此第一版可以物化 head-reduced 的 `[B,P,N4]`，但不能物化完整 per-head 大矩阵。可沿 N4 轴分 tile 计算 `[B,P,64,T]`，立刻做 ReLU、head weighted ReduceSum，再写入最终 score buffer；全部 tile 完成后调用一次普通 TopK。

#### 20.5.6 普通 TopK 优先，streaming TopK 作为后续优化

推荐 correctness 路径：

```text
score: [B,P,N4]

for n0 in range(0, N4, T):
    dot_tile = Q_index @ K_index[n0:n0+T]^T  # [B,P,64,T]
    score[..., n0:n0+T] = sum_h(w * ReLU(dot_tile))

masked_score = where(j < n4(t), score, MASK_MIN)
topk_score, topk_idx = TopK(masked_score, 512)
topk_valid = topk_idx < n4(t)
```

这样 TopK 本身就是普通静态算子，复杂点只剩 Indexer score 的分 tile/head reduction。

如果后续发现 32/64 MiB score buffer 或大轴 TopK 是瓶颈，再使用精确的 streaming Top-512。全局 Top-512 可以按候选轴分块合并，不是近似算法。设 tile 宽度为 `T`：

```text
best_score = -inf [B,P,512]
best_idx   = -1   [B,P,512]

for n0 in range(0, N4, T):              # 固定循环次数
    K_tile = K_index[:, n0:n0+T, :]     # [B,T,128]
    dot = Q_index @ K_tile^T             # [B,P,64,T]
    score_tile = sum_h(w * ReLU(dot))    # [B,P,T]
    score_tile = causal_mask(score_tile)

    merged_score = concat(best_score, score_tile)
    merged_idx   = concat(best_idx, n0 + arange(T))
    best_score, select = TopK(merged_score, 512)
    best_idx = Gather(merged_idx, select)
```

性质：

```text
Top512(A union B) = Top512(Top512(A) union B)
```

所以结果与一次性 TopK 完全相同。

streaming 版本仅优化峰值内存和大轴 TopK，不改变结果；不需要在第一版引入。

#### 20.5.7 TopK 后的 Main Cache gather

输出：

```text
topk_gather_idx: [B, P, 512]
topk_valid:      [B, P, 512]
```

从 Main Cache 得到：

```text
selected_main_kv: [B, P, 512, 512]
```

注意第二个 `P` query 维度不可省略：不同 query 的 Top-512 通常不同。这也是 CSA Prefill 不能直接把 selected KV 与 `[B,512,512]` 的 SWA pool 当成两个普通共享序列简单 concat 的原因。

Decode 的 `P=1` 没有这个 rank/共享问题：

```text
selected_main_kv_decode: [B, 512, 512]
```

### 20.6 第三刀 C：HCA128 的静态更新

HCA 没有 overlap，也没有 Indexer。其 projection 和 state 为：

```text
hca_u: [B, P, 512]
hca_z: [B, P, 512]
hca_z += APE[absolute_position % 128]

hca_u_state: [B,128,512]
hca_z_state: [B,128,512]
```

Prefill：

```text
u_cat = concat(hca_u_state, current_u)  # [B,384,512]
z_cat = concat(hca_z_state, current_z)  # [B,384,512]

aligned = Gather(cat, offset_128 + arange(256))
aligned: [B,256,512] -> [B,2,128,512]
```

对 128-token 轴逐 feature Softmax：

```text
weight = softmax(z_group, dim=128-token-axis)
candidate_hca = sum(weight * u_group, dim=128-token-axis)
candidate_hca: [B,2,512]
```

只有前：

```text
num_new_128 = (phase_128 + valid_len) // 128
```

项有效。

下一状态：

```text
hca_u_state_next = u_cat[:, valid_len : valid_len + 128]
hca_z_state_next = z_cat[:, valid_len : valid_len + 128]
```

Decode 图右侧追加 127 个 invalid row，固定 gather 128 行，最多产生 1 项：

```text
num_new_128 = (phase_128 + 1) // 128
```

只有旧 phase=127 时写入。

第 `j` 个 HCA compressed entry 的 RoPE anchor 为：

```text
compressed_pos = 128 * j
```

长期 Cache：

```text
hca_main_cache: [B,N128,512]
```

HCA 不做 TopK。query `t` 可见的 compressed 数量为：

```text
n128(t) = floor((t + 1) / 128)
```

在最大 256 Ki context 下最多为 2048 项，直接做 dense compressed Attention。

### 20.7 CSA/HCA 与 SWA 如何拼起来算 Attention

#### 20.7.1 Cache workspace 与真正的 Attention K 轴分开

Prefill 的 SWA `raw_pool` 仍可以按硬件要求保存为 512 slots，但每个 query 实际只需要固定 128 项。先为每个 query 构造窗口索引：

```text
swa_idx: [B,P,128]
```

由于 raw pool 的前 127 项固定为历史 tail，随后是 current chunk，索引甚至不依赖 `past_len`：

```text
swa_idx[q,k] = q + k
q = 0..P-1
k = 0..127
```

即 Prefill query 0 取 raw slot `0..127`，query 255 取 `255..382`；Decode query 0 同样取 `0..127`。越过序列开头、落在 PAD query/current token 的位置由 `swa_valid` 标记，不需要把负数 index 交给 Gather。Gather 后：

```text
swa_selected: [B,P,128,512]
swa_valid:    [B,P,128]
```

因此 512 是 Cache update/gather 的物理 workspace，不必成为 Attention 的 K 轴。真正的固定 Attention 宽度为：

| 图 | compressed slots | SWA slots | sink slots | 联合 logits 宽度 |
|---|---:|---:|---:|---:|
| CSA Prefill-256 | Top-512 | 128 | 1 | 641 |
| CSA Decode-1 | Top-512 | 128 | 1 | 641 |
| HCA Prefill-256 | 全部 2048 | 128 | 1 | 2177 |
| HCA Decode-1 | 全部 2048 | 128 | 1 | 2177 |

这比把 Prefill raw workspace 的 512 个位置全部送入 Attention 更直接，也让 Prefill/Decode 的单 query K 宽度一致。

#### 20.7.2 CSA：形成 `[B,P,640,512]`

TopK 后 Main Cache gather：

```text
compressed_selected: [B,P,512,512]
compressed_valid:    [B,P,512]       # 即 topk_valid
```

与 SWA gather 结果拼接：

```text
attn_kv = concat(compressed_selected, swa_selected, axis=K)
attn_kv: [B,P,640,512]

kv_valid = concat(compressed_valid, swa_valid)
kv_valid: [B,P,640]
```

对每个 query 做 query-specific MQA：

```text
Q:      [B,P,H,512]
logits: [B,P,H,640]
```

再 append 每个 head 的 sink logit：

```text
logits_with_sink: [B,P,H,641]
mask_with_sink:   [B,P,641]
```

sink slot 始终有效，但没有对应 KV value；可把它的 value 视为全 0。因此 Softmax 后只用前 640 个概率与 `attn_kv` 相乘。

这个方案会显式形成：

```text
[B,P,512,512] compressed gather
[B,P,128,512] SWA gather
```

B=1、P=256、FP16 时二者约为 128 MiB + 32 MiB。作为 correctness 静态图是清楚的；若峰值内存过高，可按 query 维分成固定小 tile，或在性能阶段做 indexed sparse-attention 算子。

#### 20.7.3 HCA：形成 `[B,P,2176,512]`

HCA 没有 TopK。把固定容量 2048 的 compressed Cache 作为逻辑候选，并按每个 query 的：

```text
hca_valid[t,j] = j < n128(t)
```

构造 mask。与 128 项 SWA gather 拼接：

```text
attn_kv: [B,P,2048+128,512] = [B,P,2176,512]
mask:    [B,P,2048+128+1]   = [B,P,2177]  # 最后一项为 sink
```

物理实现不一定要复制 2048 项 HCA Cache 到每个 query；可以让共享 HCA Cache 和 query-specific SWA 分别产生 logits，再在 2177 维上做联合 Softmax。这里的 shape 描述的是逻辑 Attention K 轴。

#### 20.7.4 不能分别 Softmax 后直接相加

正确计算必须把：

```text
raw SWA logits
compressed logits
attention sink logit
```

放在同一个归一化分母中。

可以物理 concat logits 后做一次 Softmax，也可以让两个分支输出 online-softmax statistics：

```text
branch max m_b
branch exp sum l_b
branch weighted numerator o_b
```

再合并：

```text
m = max(m_raw, m_comp, sink)
l = exp(m_raw-m)*l_raw + exp(m_comp-m)*l_comp + exp(sink-m)
o = [exp(m_raw-m)*o_raw + exp(m_comp-m)*o_comp] / l
```

attention sink 只有 denominator contribution，没有 value numerator。

Attention 完成后，还要对输出最后 64 维应用 query absolute position 的 inverse RoPE，然后再进入 grouped output projection。

### 20.8 长期 Cache 的静态图写入方式

仅仅固定 shape 并不意味着每次都复制整个 256 Ki Cache。以下做法不可接受：

```text
把 [B,65536,512] Cache 作为普通函数值完整输入、Scatter 后再完整输出
```

它可能让编译器生成整 Cache copy。

推荐将 Cache 分成两类 ABI：

#### 短状态：普通模型输入/输出

```text
SWA tail:          127 rows
CSA Main state:      8 rows
CSA Indexer state:   8 rows
HCA state:          128 rows
```

这些用 concat + fixed Gather/Slice 更新，适合作为显式静态图 Tensor。

#### 长期 Cache：持久 device buffer + 定点更新

```text
CSA Main K Cache + V Cache
CSA Indexer-K Cache
HCA Main K Cache + V Cache
```

每次图只产生固定上限的增量：

```text
CSA Prefill: 最多 64 Main + 64 Indexer entries
CSA Decode:  最多 1 + 1
HCA Prefill: 最多 2
HCA Decode:  最多 1
```

再由 in-place CacheUpdate/LLMCache 语义按：

```text
write_start
num_new
candidate_valid
```

定点写入。这里可优先复用 xhquant 的非 sliding `LLMCache` 更新语义；不要继续依赖 Python `HybridCacheTensor.cache_valid_len`。

若编译器对动态 scalar Slice/Scatter 支持不好，可让 Host 计算并传入：

```text
offset_4, write_start_4, num_new_4
offset_128, write_start_128, num_new_128
```

它们只影响 fixed-shape Gather index 和 write mask，不改变任何 Tensor shape。

### 20.9 四个完整长度例子

#### 20.9.1 第一次输入 78 token，pad 到 256

输入：

```text
past_len=0
valid_len=78
```

CSA：

```text
phase_4=0
offset_4=4
num_new_4=(0+78)//4=19
write_start_4=0
next_phase_4=2
next_offset_4=2
```

因此固定计算 64 个 candidate，但只写前 19 项。Main 与 Indexer 两个 state 都留下 token 70..77 的 projection。

HCA：

```text
phase_128=0
offset_128=128
num_new_128=0
next_phase_128=78
next_offset_128=50
```

不写 HCA Cache；128-row state 中 78 个有效项右对齐。

SWA：

```text
raw pool shape=512
只有 current 0..77 有效
next tail 中 78 项有效、前 49 项无效
```

最后一个有效 query `t=77`：

```text
n4(77)=78//4=19
n128(77)=78//128=0
```

所以 CSA Indexer 只能从 19 个 block 中选择，HCA 看不到 compressed block，但两者都有 SWA raw branch。

#### 20.9.2 第一次输入恰好 256 token

```text
CSA num_new=64，next_phase=0，next_offset=4
HCA num_new=2， next_phase=0，next_offset=128
SWA next tail=current token 129..255
```

Prefill 内每行的 compressed causal count 不同：

```text
CSA: query 3 首次看到 C4 entry 0；query 255 看到 64 项
HCA: query 127 首次看到 HCA entry 0；query 255 看到 2 项
```

#### 20.9.3 总输入 2359 token

切分：

```text
9 * 256 + 55 = 2359
```

最终：

```text
CSA entries = floor(2359/4)   = 589
CSA phase   = 2359 % 4        = 3
CSA offset  = 1

HCA entries = floor(2359/128) = 18
HCA phase   = 2359 % 128      = 55
HCA offset  = 73
```

前 9 个满 chunk 每次写 64 个 C4、2 个 HCA；最后 55-token chunk 写 13 个 C4、0 个 HCA。

#### 20.9.4 总输入 43,456 token

切分：

```text
169 * 256 + 192 = 43,456
```

最终：

```text
CSA entries = 43,456 / 4      = 10,864
CSA phase   = 0
CSA offset  = 4

HCA entries = floor(43,456/128) = 339
HCA phase   = 64
HCA offset  = 64
```

169 个满 chunk 产生：

```text
CSA 10,816 项
HCA    338 项
```

最后 192-token chunk 再产生：

```text
CSA 48 项
HCA  1 项，并留下 64-token open group
```

进入 Decode 后：

```text
CSA 再过 4 个 token 完成下一项
HCA 再过 64 个 token 完成下一项
```

#### 20.9.5 最大 256 Ki context

若 `256 Ki = 262,144 token`：

```text
N4   = 262,144 / 4   = 65,536
N128 = 262,144 / 128 =  2,048
```

静态初始化：

```text
CSA Main Cache:    [B,65536,512]
CSA Indexer Cache: [B,65536,128]
HCA Main Cache:    [B, 2048,512]
SWA tail:          [B,  127,512]
```

如果产品配置中的“256K”实际指十进制 `256,000 token`，则应从 wrap 参数重新计算为：

```text
N4   = 64,000
N128 =  2,000
```

不要在实现中硬编码 65,536/2,048。

### 20.10 推荐实现顺序

第一阶段只追求语义正确：

1. 实现统一 `past_len/valid_len/phase/offset/num_new` reference；
2. 实现 SWA `tail127 + current256 -> pool512 -> tail127`；
3. 先实现 HCA128，因为无 overlap、无 TopK；
4. 实现 CSA Main 的 `state8 + slice260 + 64 candidates`；
5. 复制同一时序实现 Indexer Compressor；
6. 用 tiled exact TopK 实现 Indexer score；
7. 实现 raw/compressed/sink 的联合 Softmax；
8. 最后再融合 mHC、Compressor、Indexer 与 CacheUpdate。

最小 correctness 矩阵：

```text
valid_len: 0,1,2,3,4,5,78,127,128,129,255,256
past phase C4:   0,1,2,3
past phase C128: 0,1,63,64,126,127
多 chunk: 257,383,2359,43456
Prefill 输出与官方一次性 reference 对齐
Prefill + Decode 输出与逐 token 官方 reference 对齐
PAD projection 不得进入 state 或 compressed Cache
TopK early history <512 时必须输出独立 valid mask；Gather index 需 clamp 到安全范围
```

本节的 offset/state 公式已用纯索引 reference 穷举：

```text
past_len = 0..520
valid_len = 0..256
C4 / C128 / SWA 各 133,897 个组合
```

检查了每个有效压缩 candidate 覆盖的绝对 token、`write_start + num_new`、以及下一 tail state；三组均通过。

### 20.11 本节依据

- [DeepSeek-V4 Technical Report：mHC、CSA、Indexer、HCA](https://arxiv.org/html/2606.19348v1)
- [vLLM DeepSeek-V4 implementation note：C4 state=8、C128 state=128、统一 256 native-token page](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md)
- 官方本地 inference：`/data01/datasets/DeepSeek-V4-Flash-0731/inference/model.py`
- 官方本地 mHC kernel：`/data01/datasets/DeepSeek-V4-Flash-0731/inference/kernel.py`
- vLLM Compressor：`/data01/home/yujy/work/vllm/vllm/models/deepseek_v4/compressor.py`
- vLLM C4/C128 fused update：`/data01/home/yujy/work/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
- vLLM Indexer：`/data01/home/yujy/work/vllm/vllm/models/deepseek_v4/attention.py` 与 `vllm/v1/attention/backends/mla/indexer.py`
- xhquant Hybrid Cache：`/data01/home/yujy/work/xhquanttool/xhquant/core/cache_tensor.py` 与 `xhquant/ops/xh/kv_cache.py`
