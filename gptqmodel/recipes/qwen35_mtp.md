# Qwen3.5 / Qwen3.6 MoE MTP W4

`quantize_qwen35_mtp` 接收原生 HF checkpoint，也接收主干和主 head 已完成
AutoRound/GPTQ 量化的 checkpoint。只补量化 MTP 非共享专家，不重新量化
主干、主 head、MTP attention、融合 FC、router、共享专家和 norm。
输入须保留浮点 MTP；旋转模型和不兼容的已打包 MTP 明确报错。
算法默认 `inherit`，复用 head 补量化的 producer 识别：AutoRound 主体使用
AutoRound，GPTQ 主体使用 GPTQ。不能把 AutoRound 的 GPTQ 存储格式识别为
GPTQ 算法。原生 HF 在整模入口继承本次算法；单独校准原生 HF 时显式指定算法。

```python
from gptqmodel.recipes.qwen35_mtp import quantize_qwen35_mtp

result = quantize_qwen35_mtp(
    model_dir="/models/Qwen3.6-main-and-head-quantized",
    mtp={"expert_bits": 4, "iters": 400},  # algorithm 默认 inherit
    calibration_jsonl="/data/pile-10k.jsonl",
    nsamples=256, seqlen=2048, device="cuda:0",
    calibration_cache="/work/mtp-calibration.pt",  # 可选
)
print(result.output_dir)
```

不指定 `output_dir` 时，在输入同级新建 `<输入目录名>_mtp_w4`。不覆盖原目录，
只重写含 MTP 非共享专家的 shard；其余张量逐项保留，主干量化配置和混合精度
分配保留。已有符合版本、位宽、分组、数值类型及实际载荷要求的 MTP 会返回
原路径，跳过数据加载和校准。

整模入口 `quantize_qwen35_autoround(..., mtp={...})` 和
`quantize_qwen35(..., mtp={...})` 在主干/head 完成后执行同一 MTP recipe。
整模入口输入是浮点 HF；已有量化产物补 MTP 使用上面的独立入口。
`mtp=None` 保持原行为，不参与主干的 `auto_mixed_precision` 搜索。

## 校准与验证

1. 用传入主模型前向得到 final-norm 后的 hidden[t]，配对 token[t+1] 的共享
   embedding。通过原浮点 MTP 的融合 FC、因果 gated attention、norm 采集专家输入。
2. 按原 router/top-k 选择专家实际输入，每个专家最多采样 8192 个 token。
   完全未被选中的专家发出 warning，记录 `bypass=true`，使用该层全局输入抽样
   做所选算法校准；不会静默回退 RTN 或把无路由样本标为实际命中。
3. AutoRound 分支联合优化 gate/up/down 的 rounding 和 min/max 参数，
   使用含 SiLU 及乘法的专家输出 MSE，默认 400 次迭代。
   GPTQ 分支采集输入 Hessian，先求解 gate/up，再用其实际打包重载的输出采集
   down 的 Hessian，执行 true-sequential GPTQ。`iters` 仅适用于 AutoRound。
   两者均为 W4、G64、`int_sym_xh`，不搜索 MTP 位宽，其他 MTP 权重留浮点供导出 RTN W8。
4. 额外读取 `mtp.validation_nsamples` 条验证文本（默认 4），不放入优化集。
   用原主 head 比较浮点 MTP 与 W4 MTP 的输出 KL、top-1 一致率，以及各自与
   主模型 next-position top-1 的匹配率。这些是 teacher-forcing 指标，不等价于
   实际投机解码接收率，也没有覆盖其他 MTP 权重的 SEFP W8 误差。

数据需至少 `nsamples + validation_nsamples` 条有效文本。来源、seed、专家
路由数量、优化误差和验证结果写入 `qwen35_mtp_recipe_provenance.json`。
可选输入缓存核对 token hash、源路径与文件版本；来源不一致直接报错。

## 权重契约

`config.json.mtp_quantization` 独立记录版本 1、`scope=routed_experts`、
`format=gptq_v2`、`expert_bits=4`、`group_size=64`、`data_type=int_sym_xh`。
`mtp.layers.0.mlp.experts.<id>.{gate,up,down}_proj` 保存
`qweight/qzeros/scales/g_idx`，采用 canonical GPTQ V2 zero-point。
不改主模型的 `quantization_config`，不要求其格式与 MTP 相同。
HF 主模型仍忽略 MTP，`Qwen35MTP.from_pretrained` 负责加载 MTP。
消费方读取原整数码和尺度，同时校验 MTP 导出位宽。

MTP 共用主模型 head；主 head 已打包时读取其真实量化载荷，不重复训练。
不能仅因存在 embedding 就把它当作 untied head 的替代权重。

## Dense MTP

Qwen3.5/3.6/3.8 dense 的单层 MTP 使用 `mtp={"block_bits": 4}`；
`block_bits` 可选 4、5、8。对 decoder block 的 attention 与 MLP Linear
逐层使用所选 AutoRound/GPTQ 算法校准，融合 FC 保留浮点供导出 W8，
norm 保留浮点，head 始终复用主模型。Dense 不使用 `expert_bits` 来划分专家；
MoE 仍使用 `expert_bits=4`，两种拓扑配置不能混用。

保存的 `mtp_quantization.scope=dense_block` 和 `block_bits` 描述真实精度，
`mtp.layers.0.{self_attn,mlp}.*_proj` 保存 GPTQ V2 载荷；W5 仍使用本地
8-bit 槽位。默认输出目录后缀为 `mtp_w<block_bits>`，已量化主干保持不变。
重载时校验实际载荷、分组和模块覆盖，不依赖输出目录名称判断量化完成。
验证指标中的 `draft_fp_to_wN_kl`/`wN_draft_main_top1_match` 随实际位宽变化。

ModelZoo unified 的 LoRA＋MTP 使用已经完成这些校准的 `quant_checkpoint`。
本 recipe 不加载或量化 LoRA 增量；多 LoRA 由导出层复用同一个量化基座。
