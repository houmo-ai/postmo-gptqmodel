# 量化回归覆盖与接入标准

量化门禁按“算法 × 数值语义 × 模型结构 × 产物行为”划分，不用两个模型名代表整个库。
本文区分运行证据、已经登记的用例和待补缺口；不是 GPTQModel 全部能力或芯片支持列表。
`cpu`、`quant` 表示执行资源，CPU 上的最小真实量化/重载同样是有效生产端证据，无需重复搬到 GPU。

## 当前证据

| 能力 | 已登记用例与边界 | 状态 |
| --- | --- | --- |
| GPTQ 基本 W4/G64，GPTQ v1/v2 保存重载 | `tests/ci/test_lm_head_requantization.py` 的最小 Qwen2 主体真实量化；sym=True、desc_act=False | 前轮 CPU 通过；不代表所有 group_size/desc_act |
| GPTQ / AutoRound dense W4/W8 G64 | `test_quantization_guard.py`，Qwen3.5 文本路径，两种 attention 都实际量化；AutoRound 默认 `int_sym_gptq`，有限校准、packed/配置与重载前向检查 | 本轮 4 组 GPU 通过，连同 42 项 policy 共 46 项、约 228 秒；这是产物/重载 smoke，不是质量阈值 |
| XH AutoRound 混合精度、hybrid attention、MoE、MTP | `tests/ci/test_qwen3_next_autoround.py::test_real_autoround_quantize_save_load`；`int_sym_xh`，组合 `(MTP,attention bits,expert bits)` = `(否,8,4)`、`(是,8,4)`、`(否,16,8)`；保存重载载荷与 logits 一致 | 前轮 3 组 GPU 通过；不代表 GPTQ MoE 或完整 VLM |
| Ling KDA/MLA/MoE、MLA 吸收、MTP 保留 | 原始布局 AutoRound（MTP 0/1）；吸收布局 GPTQ/AutoRound；W8 attention / W4 routed experts、G64，实际量化、整数载荷保存重载、MLA prefill/decode 一致 | 4 组 GPU 通过，连同 policy 共 46 项、约 135 秒；使用真实 remote code 的缩小维度模型，非原尺寸权重 block |
| W4/W5 head 的 GPTQ/AutoRound 补量化 | `tests/ci/test_lm_head_requantization.py`；W5 使用 W8 存储槽位，检查逻辑位宽、tied head、主体保留、packed 复用和错误参数 | 前轮 CPU 通过；不代表所有 expert 的 W5 |
| W4 打包饱和 | `tests/ci/test_pack_clamping.py`；检查越界值 clamp 后再 pack，避免回绕 | 前轮 CPU 通过；不代表所有硬件 kernel |
| 产物消费 | ModelZoo 的 head / XH dtype / Ling MLA 用例 | 前轮通过；不能替代生产端完整量化 |

“前轮”指 README 中的验证基线。5 种复用/自切浮点 block 的来源与结构见
[fixture 说明](fixtures/README.md)；它们通过浮点前向不等于对应算法量化已通过。
格式标签 `quant_method=gptq` 不足以判断生产算法；`int_sym_gptq` 与 `int_sym_xh` 要分别验证。

## 应优先补齐的范围

| 优先级 | 补充场景 | 为什么单独保留 | 接入前提 |
| --- | --- | --- | --- |
| P0 | dense AutoRound 的 XH 数值网格补充代表 | 当前 dense 基准是 GPTQ 网格，XH 证据来自 Qwen3-Next | 复用 2 层 dense block，明确数值类型/格式，实际量化/保存/重载 |
| P0 | 扩充原尺寸 GPTQ MoE 代表 | 已有 Ling 缩小维度 MoE 的 GPTQ 证据；仍需原尺寸 expert 遍历、校准覆盖与数值基线 | 2 层 Qwen MoE 和 3 层 Ling block 已就绪；检查路由与 expert 覆盖 |
| P1 | Ling 原尺寸权重与 KDA cached decode 专项 | 当前 4 组量化覆盖缩小维度；量化校准执行 KDA，但缓存 decode 的显式一致性断言针对 MLA | [3 层真实 block](fixtures/ling3-flash.md) 已通过 shape/BF16 前向，原尺寸量化和 KDA decode 另补 |
| P1 | Gemma 的特殊层输入/shared KV、模型特有 rotation/MTP | recipe 签名正确不证明层遍历、激活采集与数学变换正确 | 4 层 Gemma shared-KV / 2 层 Gemma MoE 已就绪；rotation/MTP 仍需专用用例 |
| 按功能接入 | 新 VLM vision/audio 量化、新逻辑位宽、desc_act、非对称、不同 group_size、offload/resume | 每项都会改变实际量化或产物行为 | 变更实现的同一提交增加代表回归，避免提前展开所有维度的笛卡尔积 |

新 family 完全复用现有量化机制时，增加本 family 模块树、参数、校准入口等真实功能用例，
可以复用已有算法代表集；若引入不同结构、算法执行或载荷布局，则必须新增最小真实量化用例。
同一架构的新参数规模不必重复全量训练；不同机制不能只因同属一个 family 就省略回归。
如 MiniCPM5 当前登记的是定义/模块树测试，尚不构成其 recipe 的完整行为或量化证据。

## 每个真实量化用例的最低断言

1. 输入与配置可追溯，实际执行了要求的后端；量化模块被正确选中，要求保留浮点的 gate/norm 等未被误量化。
2. 校验实际 qweight/qzeros/scales/g_idx（按格式要求）、shape、dtype、位宽/逻辑位宽、分组和有效 scale。
3. 使用真实 loader 重载，检查相关 metadata、辅助载荷与模块结构；执行有限输出检查。
4. 以保存前后相同 dtype 的输出/整数载荷或明确参考验证一致性；仅有限 logits 是 smoke，不是量化质量阈值。
5. 原 checkpoint 不覆盖，失败不能发布成功记录；按变更需要覆盖非法参数、packed 复用、tied head 等边界。

用例旁边只加自己的 `.impact.json`，列出确实执行的 recipe/模型/helper。共享实现修改选择
所有声明的使用方；独立新功能不改旧声明。通过 CPU 小模型可验证的行为不重复占用 GPU。
大模型 PPL/任务精度、长序列、多卡、性能和各推理 kernel 的完整矩阵由专项/定时验证承担，
其任务配置交给 CI 同事；不能把未运行的专项写成门禁已覆盖。
