# EBSS Native Generation

本目录实现了仓库内原生的 EBSS（Expert-Balanced Beam Search）校准集生成逻辑。

目标：

- 将 MoE 校准集生成逻辑从旧的脚本 + 全局 monkey patch 方案收敛到仓库内模块
- 让 EBSS 的数学逻辑、模型适配逻辑、beam search 状态逻辑、provider/runtime 接入逻辑清晰分层
- 支持“直接生成内存中的 calibration rows”，而不是必须先落盘 JSONL 再回读

## 1. EBSS 的核心数学逻辑

### 1.1 专家累计计数

对第 $l$ 层、第 $b$ 个 beam、第 $e$ 个 expert，定义累计计数：

$$
c_{l,b,e}
$$

表示该 beam 在解码到当前 step 为止，命中该 expert 的累计次数。

每个 step 都会从模型适配器中拿到当前 step 的专家命中矩阵：

$$
\Delta c_{l,b,e}
$$

累计更新：

$$
c_{l,b,e} \leftarrow c_{l,b,e} + \Delta c_{l,b,e}
$$

对应代码：

- [ebss_score.py](gptqmodel/quantization/calibration/generation/ebss_score.py) 中的 `update_cumulative_expert_counts`
- [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py) 中的 `EBSSBeamSearchState.update_from_step_counts`

### 1.2 专家均衡分数

对每层每个 beam，在 expert 维度上计算方差：

$$
v_{l,b} = \mathrm{Var}_e(c_{l,b,e})
$$

方差越大，表示该 beam 的专家使用越集中，不够均衡。

再对所有有效层求平均：

$$
s_b = \frac{1}{L} \sum_{l=1}^{L} v_{l,b}
$$

再在 beam 维度归一化：

$$
\hat{s}_b = \frac{s_b}{\sum_j s_j}
$$

仓库里也支持“频率模式”，即先将累计计数归一化成专家访问频率：

$$
p_{l,b,e} = \frac{c_{l,b,e}}{\sum_e c_{l,b,e} + \epsilon}
$$

再对 $p_{l,b,e}$ 计算方差。相比直接对计数算方差，频率模式对句长增长更稳定。

对应代码：

- [ebss_score.py](gptqmodel/quantization/calibration/generation/ebss_score.py) 中的 `compute_beam_variance_scores`
- [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py) 中的 `EBSSBeamSearchState.compute_expert_scores`

### 1.3 将 EBSS 作为 beam penalty

当前仓库使用 log-score 形式的惩罚：

$$
\text{beam\_score}_b \leftarrow \text{beam\_score}_b - \lambda \hat{s}_b
$$

其中：

- $\lambda$ 是惩罚强度，对应 `penalty_strength`
- 分数越大表示专家越不均衡，因此会被减掉更多

将其并入 token 的对数概率：

$$
\log P(\text{token} \mid b) + \text{beam\_score}_b
$$

对应代码：

- [ebss_score.py](gptqmodel/quantization/calibration/generation/ebss_score.py) 中的 `apply_ebss_penalty` 与 `apply_ebss_penalty_to_log_probs`
- [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py) 中的 `EBSSBeamSearchState.apply_penalty_to_log_probs`

## 2. 代码分层

### 2.1 纯算法层

- [ebss_score.py](gptqmodel/quantization/calibration/generation/ebss_score.py)

职责：

- 更新累计专家计数
- 处理 beam reorder 后的状态对齐
- 计算 expert variance score
- 将 penalty 应用到 beam score / log_probs

这一层不依赖具体模型结构。

### 2.2 模型适配层

- [expert_stats_adapter.py](gptqmodel/quantization/calibration/generation/expert_stats_adapter.py)
- [adapters/qwen3_moe.py](gptqmodel/quantization/calibration/generation/adapters/qwen3_moe.py)
- [adapters/qwen3_next.py](gptqmodel/quantization/calibration/generation/adapters/qwen3_next.py)
- [adapters/registry.py](gptqmodel/quantization/calibration/generation/adapters/registry.py)

职责：

- 将不同模型的 router / sparse moe block 暴露成统一接口
- 管理统计开关
- 读取每个 step 的 expert 统计
- 清理 step 统计
- 在需要时跟随 beam reorder 做状态同步

统一接口：

- `attach()`
- `enable_collection()`
- `disable_collection()`
- `get_step_counts()`
- `clear_step_counts()`
- `reorder_state()`
- `detach()`

### 2.3 Beam 状态层

- [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py)

职责：

- 持有 `cumulative_counts`
- 每个 step 从 adapter 更新累计计数
- 在 beam 重排时保持统计与 beam 对齐
- 计算当前 expert score
- 将 penalty 注入到 log_probs
- 将 `_beam_search` 只绑定到模型实例，而不是全局改写 transformers

### 2.4 Provider 层

- [moe_ebss_provider.py](gptqmodel/quantization/calibration/generation/moe_ebss_provider.py)

职责：

- 根据 `QuantizeConfig.calibration_generation` 驱动原生 EBSS 生成
- 选择具体 model type 和 adapter
- 绑定实例级 beam search
- 生成内存 rows
- 可选导出 JSONL

### 2.5 Runtime 层

- [runtime.py](gptqmodel/quantization/calibration/generation/runtime.py)

职责：

- 当量化启用校准集自生成时，调用 provider
- 优先直接接收 provider 返回的内存 rows
- 仅在需要时回退为 `output_path -> JSONL loader`

## 3. 执行流程

### 3.1 量化入口

`BaseQModel.quantize()` 会在量化前调用 runtime：

1. 读取 `quantize_config.calibration_generation`
2. 若开启，则获取 provider
3. provider 原生生成 calibration rows
4. rows 直接传给 `prepare_calibration_dataset`
5. 后续量化流程保持不变

### 3.2 Provider 内部流程

以 `MoeEbssProvider.generate()` 为例：

1. 解析 `model_type` 和 `model_path`
2. 加载 generation model
3. 创建对应 expert stats adapter
4. 将旧的 `_beam_search_ebss` 绑定到模型实例 `_beam_search`
5. 采样若干起始 token
6. 执行 beam generation
7. decode 为文本 rows
8. 返回 `{rows, output_path}`

注意：

- 当前仍复用了旧的 `_beam_search_ebss` 主循环实现
- 但已不再对 `transformers.GenerationMixin` 做全局 monkey patch
- 绑定范围被限制在单个 generation model 实例上

## 4. 当前与旧实现的关系

当前 generation 目录已经承载了 EBSS 的主数学逻辑与仓库接入逻辑，但 beam search 主循环仍沿用了旧的：

- [../moe_ebss/custom_ebss.py](gptqmodel/quantization/calibration/moe_ebss/custom_ebss.py)

原因：

- 这样可以先把全局 monkey patch 替换为实例级绑定
- 同时不在一次改动中重写整段 Hugging Face beam search 主循环，风险更低

后续若继续演进，建议下一步是把 `custom_ebss.py` 中的 `_beam_search_ebss` 正式迁移到本目录，彻底完成 beam search 本地化。

## 5. 文件职责总览

- [base.py](gptqmodel/quantization/calibration/generation/base.py)
  provider 基类与注册表
- [jsonl_loader.py](gptqmodel/quantization/calibration/generation/jsonl_loader.py)
  JSONL 回读
- [ebss_score.py](gptqmodel/quantization/calibration/generation/ebss_score.py)
  纯数学逻辑
- [expert_stats_adapter.py](gptqmodel/quantization/calibration/generation/expert_stats_adapter.py)
  统一适配器接口
- [adapters/qwen3_moe.py](gptqmodel/quantization/calibration/generation/adapters/qwen3_moe.py)
  Qwen3-MoE 适配器
- [adapters/qwen3_next.py](gptqmodel/quantization/calibration/generation/adapters/qwen3_next.py)
  Qwen3-Next 适配器
- [adapters/registry.py](gptqmodel/quantization/calibration/generation/adapters/registry.py)
  适配器注册表
- [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py)
  beam 状态管理与实例级绑定
- [moe_ebss_provider.py](gptqmodel/quantization/calibration/generation/moe_ebss_provider.py)
  原生 EBSS provider
- [runtime.py](gptqmodel/quantization/calibration/generation/runtime.py)
  量化入口运行时接入

## 6. 当前状态结论

本目录现在已经具备：

- 完整的 EBSS 数学逻辑
- 清晰的模型适配层
- 原生 provider / runtime 集成
- 进程内 rows 直通量化 pipeline
- 可选 JSONL 导出能力

仍保留的兼容依赖只有一项：

- beam search 主循环本体仍暂时复用旧的 `custom_ebss.py`

这不影响当前功能闭环，但如果你要把 EBSS 完整收口到本目录，下一步应当把 `_beam_search_ebss` 正式迁移到 [ebss_beam_search.py](gptqmodel/quantization/calibration/generation/ebss_beam_search.py)。