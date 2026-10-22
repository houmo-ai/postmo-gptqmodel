# Qwen3-VL 特征编码器量化

公开入口为 `from gptqmodel.recipes import quantize_qwen3_vl`，算法通过
`method="gptq"` 或 `method="autoround"` 选择。GPTQ 使用原生对称网格，AutoRound 使用 `int_sym_xh`；两者均为 G64，
只量化 `feature_layer` 之前的语言层；ViT、未使用的语言层和 LM head 保留浮点。
W4A8 中的 A8、视觉 W8A8 由 ModelZoo/SDK 部署阶段负责，本 recipe 不生成 HMONNX。

```python
import torch
from gptqmodel.recipes import quantize_qwen3_vl

result = quantize_qwen3_vl(
    model_dir="/models/MiniMax-H3/text_encoder",
    tokenizer_dir="/models/MiniMax-H3/tokenizer",
    processor_dir="/models/MiniMax-H3/processor",
    output_dir="/outputs/qwen3-vl-w4",
    calibration_data=torch.load("/calibration/h3.pt", weights_only=True),
    method="autoround", feature_layer=50, bits=4, group_size=64,
    device="cuda:0", seed=42, iters=200,
)
print(result.output_dir, result.provenance)
```

`calibration_data` 是原生 processor 输出的 tensor 字典列表，每项 batch=1，保留
`input_ids`、`attention_mask`、图像/视频像素、grid 和 `mm_token_type_ids`。
不添加 chat template，不截断，不将多模态样本替换为纯文本。纯文本、图像和视频应按实际
部署输入分布准备；使用真实 H3 presentation，包括视觉标签和占位 token。
`feature_layer` 必须小于原始语言层数，对应原生 `hidden_states[feature_layer]`，不含最终 norm。

GPTQ 逐层重放会在进入下一层前加回原生 DeepStack 增量；AutoRound 在前三次注入边界分别
缓存，剩余语言层共用一个连续重放分组，仍按默认 `nblocks=1` 逐层优化。
文本与视觉样本的可选位置参数可能在 tensor 和 `None` 之间变化；AutoRound 的 batch-one
缓存保留每条样本的占位，避免位置参数与隐藏状态错位。回归包含交替模态及变长输入。
`iters` 只影响 AutoRound，GPTQ 不使用该参数。
完整命令行入口为 `example_qwen3_vl_features.py --help`。

输出目录必须尚不存在，父目录必须存在。保存后通过真实 GPTQModel loader 重载并运行
原生特征前向，成功后原子发布目录及 `qwen3_vl_feature_quantization.json`。
记录含算法、位宽、分组、seed、校准条数和长度、特征层、DeepStack 策略和重载结果。
这个检查验证产物可执行，不代表大模型视频生成质量已经达标。

回归在 `tests/ci/test_qwen3_vl_calibration.py` 和 `test_qwen3_vl_recipe.py`；
ModelZoo 消费回归为 `test_sol_h3_conditioner_quantization.py`，执行两种算法到
SDK W4A8 图导出的真实链路。必须使用本仓随附的 AutoRound，包含 dtype-only dynamic
override 修复；旧安装版本可能错误地将已量化层标成浮点。
