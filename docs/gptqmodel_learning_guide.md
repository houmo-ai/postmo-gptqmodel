# GPTQModel 学习路线与代码调用链说明

这份文档面向第一次系统阅读 `gptqmodel` 代码的人，目标是回答四个问题：

1. 从示例脚本开始，代码到底是怎么一路走到量化执行的。
2. `GPTQModel`、`BaseQModel`、`ModuleLooper`、`module_tree` 分别负责什么。
3. `defuser.convert_model()` 和 `nn_modules/converter.py` 里的 converter 有什么区别。
4. Qwen3 / Qwen3-MoE / Qwen3-Next 这类 packed experts 模型，当前仓库是怎么处理的，还缺什么。

本文以 `examples/quantization/examples/example_qwen3_next.py` 为主线，也适用于大多数 GPTQModel 支持的文本模型。

## 建议阅读顺序

如果想最快建立整体认知，建议按下面顺序读代码：

1. `examples/quantization/examples/example_qwen3_next.py`
2. `gptqmodel/models/auto.py`
3. `gptqmodel/models/loader.py`
4. `gptqmodel/models/base.py`
5. `gptqmodel/looper/module_looper.py`
6. `gptqmodel/looper/stage_layer.py`
7. `gptqmodel/looper/stage_subset.py`
8. `gptqmodel/models/definitions/qwen3_next.py`
9. `gptqmodel/models/moe_lifecycle.py`
10. `gptqmodel/nn_modules/converter.py`

这样阅读时，先看“流程”，再看“模型适配细节”，不容易陷在某个模型 definition 里。

## 一、全局架构

从职责划分看，核心模块可以理解为：

- `examples/...`：准备参数、数据集、`QuantizeConfig`，调用 `GPTQModel`。
- `models/auto.py`：根据 `model_type` 选择具体模型 definition。
- `models/loader.py`：负责加载未量化模型或已量化模型。
- `models/base.py`：`BaseQModel` 的主体逻辑，负责：
  - 吃掉 `module_tree`
  - 构造量化流程
  - 调起 `ModuleLooper`
- `looper/module_looper.py`：量化执行总调度器。
- `looper/stage_layer.py`：按层执行量化。
- `looper/stage_subset.py`：按层内 subset 执行前向、统计、量化。
- `models/definitions/*.py`：每种模型的适配定义。
- `models/moe_lifecycle.py`：MoE 特殊前向逻辑，主要用于“把校准数据送到 experts”。
- `nn_modules/converter.py`：少数模型在量化阶段需要做的结构重写。

可以用下面这张图理解：

```text
example script
  -> GPTQModel.load(...)
    -> choose model definition
    -> load HF model
    -> BaseQModel(...)
  -> model.quantize(...)
    -> ModuleLooper(...)
    -> cache inputs
    -> build layer subsets from module_tree
    -> run layer stage
      -> optional per-layer converter
      -> hook modules
      -> run subset stage
        -> forward
        -> collect stats
        -> quantize/pack
  -> model.save(...)
```

## 二、从 example 到 `GPTQModel.load`

### 1. 示例脚本是入口

以 `examples/quantization/examples/example_qwen3_next.py` 为例，脚本主要做三件事：

- 解析命令行参数
- 构造 `QuantizeConfig`
- 调用：
  - `GPTQModel.load(...)`
  - `model.quantize(...)`

这个文件值得先看三段：

- `build_moe_config()`：MoE routing 配置
- `build_dynamic_quant_config()`：按模块名分配位宽
- `main()`：从加载到量化的主流程

### 2. `GPTQModel.load(...)` 只是分发入口

对应文件：`gptqmodel/models/auto.py`

关键函数：

- `GPTQModel.load(...)`
- `GPTQModel.from_pretrained(...)`
- `GPTQModel.from_quantized(...)`
- `check_and_get_model_definition(...)`

职责：

- 读取 HF config
- 判断当前路径指向的是未量化模型还是已量化模型
- 根据 `config.model_type` 选择对应的 definition class

例如：

- `qwen3_next -> Qwen3NextGPTQ`
- `qwen3_moe -> Qwen3MoeQModel`
- `llama4 -> Llama4QModel`

也就是说，`auto.py` 的角色更像“总路由器”，不是具体的加载器。

## 三、真正加载模型的是 `models/loader.py`

### 1. 入口：`ModelLoader.from_pretrained(...)`

对应文件：`gptqmodel/models/loader.py`

这是通过 mixin 的方式挂到 `BaseQModel` 上的。它负责：

1. 读 config
2. 调模型定义的 `before_model_load(...)`
3. 调 HF `from_pretrained(...)` 或 `from_config(...)`
4. 跑一遍 `defuser.convert_model(...)`
5. 构造 `BaseQModel` 实例

### 2. `before_model_load(...)`

这是模型 definition 的“加载前钩子”。

默认定义在 `gptqmodel/models/base.py` 中，什么都不做。具体模型可以覆盖它。

Qwen3-Next 现在就在这里做了一件重要事情：

- 默认禁用 FLA fast path
- 避免 `transformers 5.3.x` 下的 FLA/Triton 路径影响量化稳定性

对应文件：

- `gptqmodel/models/definitions/qwen3_next.py`

### 3. `defuser.convert_model(...)` 是加载期的通用转换

这个调用在 `loader.py` 里出现了三次：

- 加载未量化模型时
- 加载壳模型时
- 加载量化模型时

它的角色是：

- 对模型做通用结构转换、兼容性处理、替换部分模块

但要注意：

- 它不是 `nn_modules/converter.py` 里的 model-specific converter
- 它不会自动把 Qwen3-Next 的 packed experts 变成 `experts.0.gate_proj` 这种结构

这是理解后面 MoE 适配问题的关键。

## 四、`BaseQModel` 是什么

对应文件：`gptqmodel/models/base.py`

`BaseQModel` 是整个项目最重要的类之一。加载完成后，最终拿到的对象其实就是某个 model definition 继承 `BaseQModel` 后得到的实例。

### 1. 构造时做了什么

`BaseQModel.__init__(...)` 里最重要的几件事：

- 处理 `module_tree`
- 应用 `module_tree_overrides`
- 如果没有 `module_tree`，尝试 auto-detect
- 保存 `quantize_config`
- 调用 `after_model_load(...)`

### 2. `after_model_load(...)`

这是“加载后钩子”。

默认实现仍然是不做额外操作，但某些模型会在这里做：

- 特殊结构替换
- Processor 初始化
- 模型对象修补

### 3. `module_tree` 是什么

`module_tree` 是 GPTQModel 里非常核心的抽象。

它用来描述：

- 一个 transformer layer 里有哪些子模块需要被量化
- 这些子模块在 forward 执行中大致是什么顺序
- 哪些模块不量化，只做 capture 或辅助

例如 Qwen3-Next 的定义在：

- `gptqmodel/models/definitions/qwen3_next.py`

它描述了：

- `input_layernorm`
- `self_attn`
- `linear_attn`
- `post_attention_layernorm`
- `mlp`
- `mlp:moe`

### 4. `module_tree` 语法速记

常见标记：

- `"#"`：层列表占位符
- `:0`, `:1`：同一组中的执行顺序 / alias bucket
- `:!`：存在但不量化
- `:?`：可选模块
- `:moe`：标记这是 MoE block

先把它理解成“量化子图描述语言”就够了。

## 五、`module_tree` 什么时候真正生效

不是在加载时立刻展开，而是在量化时。

### 1. 入口：`BaseQModel.quantize(...)`

对应文件：`gptqmodel/models/base.py`

这个函数主要负责：

- 检查量化配置
- 选择 quant kernel
- 构建 processors
- 创建 `ModuleLooper`
- 调 `module_looper.loop()`

### 2. `simple_layer_modules(...)`

它会把 `module_tree` 变成“每层的模块子集列表”。

主要步骤：

1. `build_layer_modules(cls.module_tree, ...)`
2. `build_moe_modules_if_need(...)`
3. `filter_not_quantize_module(...)`

这一步之后，`module_tree` 才真的变成：

- 第 0 个 subset 量化哪些模块
- 第 1 个 subset 量化哪些模块
- 哪些是 capture-only

所以，如果你要调试“某个 definition 写得对不对”，最终一定要看：

- 它展开后的 `simple_layer_modules(...)` 结果

## 六、`ModuleLooper` 是量化执行总调度器

对应文件：`gptqmodel/looper/module_looper.py`

这是量化阶段的核心 orchestrator。

### 1. 它负责什么

- 管理 processors
- 缓存输入
- 按层遍历
- 按 subset 执行前向与量化
- 管理多设备线程池
- 管理 MoE routing override / bypass

### 2. 主入口：`ModuleLooper.loop()`

主流程：

1. `cache_inputs(...)`
2. 生成 `layer_modules`
3. `run_layer_stage(...)`

### 3. 输入缓存阶段

通过 `StageInputsCapture.cache_inputs(...)` 完成。

作用：

- 在还没有真正量化层之前
- 先把校准集一路 forward
- 把每层输入缓存起来

这样后续量化每一层时，不用每次都从 embedding 重跑全模型。

## 七、`stage_layer.py` 是按层执行的核心

对应文件：`gptqmodel/looper/stage_layer.py`

这是理解 converter 调用时机的关键文件。

### 1. `run_layer_stage(...)`

它的粒度是“每一层”。

每处理一层，大概会做：

1. 取出当前 layer module
2. 跑 `pre_quantize(...)`
3. 如果当前模型在 `MODULE_CONVERTER_MAP` 里，先跑一次 converter
4. 把当前层的可量化模块替换成 hookable 版本
5. 根据 `module_tree` 展开出来的 subsets，逐个执行 `run_subset_stage(...)`

### 2. 这里的 converter 是什么

这里调用的是：

- `gptqmodel/nn_modules/converter.py`

具体触发逻辑：

```python
model_type = looper.gptq_model.model.config.model_type
if model_type in MODULE_CONVERTER_MAP:
    converter = MODULE_CONVERTER_MAP[model_type]
    module = converter(module, ...)
```

所以它是：

- 量化阶段
- 按层调用
- 按 `model_type` 条件触发

### 3. 当前哪些模型真的接了这个链路

现在 `MODULE_CONVERTER_MAP` 里接入的主要是：

- `llama4`
- `gpt_oss`
- `glm4v`

而下面这些虽然有 converter 模板，但没有接入：

- `qwen3_5_moe`
- `qwen3_moe`
- `qwen3_next`

这也是为什么你看到 `Qwen3_5MoeExpertsNew`，但实际跑代码时它并没有生效。

## 八、`stage_subset.py` 是按子集执行量化

对应文件：`gptqmodel/looper/stage_subset.py`

这个文件负责“层内 subset”的真正执行。

职责包括：

- 给 subset 挂临时 forward hooks
- 调 forward
- 收集输入激活 / 统计量
- 检查哪些模块没有被触发
- 调 processor 的量化逻辑

这里也是 MoE 问题最容易暴露的地方：

- 如果某些 expert 根本没有被路由到
- 那么它们就拿不到 calibration activation
- 最后会被判定为“未被调用”
- 可能被跳过或加入动态排除

## 九、MoE 相关逻辑怎么分层看

理解 MoE 建议分三层看：

### 1. definition 层

对应文件：

- `gptqmodel/models/definitions/qwen3_next.py`
- `gptqmodel/models/definitions/qwen3_moe.py`
- `gptqmodel/models/definitions/qwen2_moe.py`

这层负责描述：

- 哪个 block 是 MoE
- `shared_expert`、`gate`、`experts` 怎么命名
- 是否有 `dynamic_expert_index`

### 2. lifecycle hooks 层

对应文件：

- `gptqmodel/models/moe_lifecycle.py`

这层负责：

- 在 bypass 模式下
- 临时替换 MoE block 的 forward
- 把 hidden states 主动送到 experts

### 3. looper 层

对应文件：

- `gptqmodel/looper/module_looper.py`
- `gptqmodel/looper/stage_subset.py`

这层负责：

- routing override
- routing bypass
- expert subset 分组
- VRAM 控制

## 十、`defuser.convert_model()` 和 `nn_modules/converter.py` 的区别

这是最容易混淆的地方。

### 1. `defuser.convert_model()`

调用时机：

- 模型加载阶段

位置：

- `models/loader.py`

角色：

- 通用结构转换
- 统一兼容处理

特点：

- 对很多模型都可能生效
- 不是针对单个 model_type 的量化期重写器

### 2. `nn_modules/converter.py`

调用时机：

- 量化阶段
- 每层处理前

位置：

- `looper/stage_layer.py`

角色：

- 少量模型的结构重写
- 让当前 layer 更适合 GPTQ 的量化逻辑

典型用途：

- 把 packed experts 转成逐 expert 的 `Linear` 结构

### 3. 一句话记忆

- `defuser.convert_model()`：加载时的通用预处理
- `MODULE_CONVERTER_MAP`：量化时的模型专用重写

## 十一、Qwen3 / Qwen3-MoE / Qwen3-Next 的 packed experts 问题

这是当前学习和适配时最重要的现实问题。

### 1. HF 里的真实结构

Qwen3-MoE 和 Qwen3-Next 的非共享 experts 都是 packed 结构：

- `experts.gate_up_proj`
- `experts.down_proj`

不是：

- `experts.0.gate_proj`
- `experts.0.up_proj`
- `experts.0.down_proj`

而 `shared_expert` 则通常还是标准 MLP：

- `shared_expert.gate_proj`
- `shared_expert.up_proj`
- `shared_expert.down_proj`

### 2. 这会导致什么问题

如果 definition 仍然按：

- `experts.#.gate_proj`
- `experts.#.up_proj`
- `experts.#.down_proj`

来写 `module_tree`，那么语义上是“想按 expert 分开量化”，但和 HF 原始对象结构不一致。

这时就会出现三种情况：

1. 名字根本找不到
2. MoE lifecycle hooks 无法正确枚举 experts
3. 动态 bits 和 subset 划分无法准确命中非共享 experts

### 3. 当前仓库里已有的模板

`gptqmodel/nn_modules/converter.py` 里已经有一个可复用模板：

- `Qwen3_5MoeExpertsNew`

它做的事情是：

- 把 `gate_up_proj[i]` 拆成 `gate_proj` 和 `up_proj`
- 把 `down_proj[i]` 拆出来
- 重建成 `experts[i].gate_proj/up_proj/down_proj`

这正是 Qwen3-MoE / Qwen3-Next 也应该走的方向。

### 4. 当前状态

现状可以简化成一句话：

- 仓库里已经有“怎么拆 packed experts”的模板
- 但 Qwen3-MoE / Qwen3-Next 还没把这条链接起来

## 十二、`Qwen3_5MoeExpertsNew` 为什么看起来存在，但没效果

因为它现在属于“实现了，但没注册”。

具体表现：

- converter 函数已经写好
- `MODULE_CONVERTER_MAP` 中没有启用它
- `Qwen3.5-MoE` 的 definition 也没有主动在加载后替换掉原始 experts

所以你如果单看 `converter.py`，会以为仓库已经支持了这类 packed experts 拆分；但从真实执行链看，它目前仍然只是模板。

## 十三、如果要继续研究，建议怎样调试

最有效的调试顺序是：

### 1. 先确认 model_type

看：

- `config.model_type`
- `auto.py` 中映射到哪个 definition

### 2. 再确认 `module_tree`

看：

- 当前 definition 里的 `module_tree`
- `simple_layer_modules(...)` 展开后的结果

### 3. 再确认原始 layer 真实结构

对比：

- `layer.named_modules()`
- `layer.named_parameters()`

看它到底是：

- packed experts
- 还是逐 expert 的 `ModuleList`

### 4. 最后确认 converter 是否被调用

看：

- `stage_layer.py` 中是否命中 `MODULE_CONVERTER_MAP`
- converter 运行后当前 layer 结构是否发生改变

这是定位“definition 写得不对”还是“converter 没接上”的最快方法。

## 十四、推荐的学习方式

如果只想建立清晰认知，不建议一开始读完整个 `base.py`。更好的方式是：

1. 先从 example 跑一遍
2. 顺着 `GPTQModel.load(...)` 看加载链
3. 顺着 `model.quantize(...)` 看执行链
4. 只在遇到具体问题时，再回到模型 definition 或 MoE lifecycle

最常见的误区是：

- 一开始就盯着 `module_tree`
- 但还没搞清它什么时候被展开、什么时候被实际使用

## 十五、当前关于 Qwen3-Next 的结论

截至目前，对 Qwen3-Next 可以先记住这几点：

- `shared_expert` 是标准 MLP，容易适配
- 非共享 `experts` 是 packed tensor，不是逐 expert 子模块
- `module_tree` 里写成 `experts.#.gate_proj/up_proj/down_proj`，语义方向是对的，但和原始 HF 对象结构不一致
- 真正稳妥的方案不是硬改 `module_tree` 去适配 packed 参数，而是先加 converter，把 experts 改造成逐 expert 模块结构

## 十六、后续可以继续补充的文档主题

这份文档更偏“总览”。如果后面继续深入，建议再单独写三份补充文档：

1. `module_tree` 语法与展开规则说明
2. `ModuleLooper / stage_layer / stage_subset` 量化执行细节
3. MoE packed experts 适配方案设计说明

这三份文档拆开后，会比把所有细节都塞在一个文件里更容易维护。
