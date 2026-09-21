# GLM-5.3-Flash 混合精度 GPTQ recipe

当前 ModelZoo 部署默认改用 [AutoRound recipe](glm5_3_flash_autoround.md)。
本文仅描述保留的 GPTQ 生产端入口。

公开入口是 `gptqmodel.recipes.glm5_3_flash.quantize_glm5_3_flash`，输入完整的
`glm5_next` 图文/视频模型，保留视觉编码器，输出 `gptqmodel_hf` checkpoint。
固定 GPTQ、对称整数、`desc_act=False`、G64：数字编号的非共享专家为 W4，
其余纳入 GPTQ 的 Linear 为 W8。不把 RTN 或 AutoRound 产物标为 GPTQ。

| 部件 | GPTQ checkpoint | ModelZoo 部署 |
| --- | --- | --- |
| `mlp.experts.<数字>.{gate,up,down}_proj` | W4 G64 | W4 G64 A16 |
| KDA/DSA 普通投影、dense MLP、shared experts | W8 G64 | W8 A16 |
| KDA `forget_gate.f_a_proj/f_b_proj`、`b_proj` | 保持浮点 | W16 A16 |
| vision、head、router、indexer、mHC、conv、DSA `kv_b_proj` 因子 | 保持浮点 | 导出时一次 W8 RTN |

DSA 保留 HF 未吸收布局。ModelZoo 把浮点 `kv_b_proj` 拆分、转置为 K/V 因子后
各量化一次，不对已量化权重做矩阵乘积后再量化。MoE 校准遍历全部专家；down-proj
输入使用原生 `_apply_gate`，保留 gate/up clamp，不以普通未裁剪 SwiGLU 替代。
仅支持当前发布配置的独立 `full` indexer；MTP 张量不进入 45 层推理主干。

```bash
conda activate xhquant_55_tf_latest_20260828
# 在调度系统分配的四张卡内运行；下例编号按实际空闲情况替换。
export CUDA_VISIBLE_DEVICES=1,4,5,6
python examples/quantization/examples/example_glm5_3_flash.py \
  --model /path/to/GLM-5.3-Flash --out /path/to/glm53-gptq-w4g64 \
  --prepared-model-dir /path/to/glm53-bf16-input \
  --preparation-devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --calibration-jsonl /path/to/NeelNanda-pile-10k.jsonl
```

准备阶段按原始 shard 流式解码 block-FP8，可跨 shard 读取 scale，保留原生 FP32
敏感参数；不会覆盖源权重。BF16 中间 checkpoint 约 640 GB，完整量化会在主机中加载
模型，需预留相应 RAM；低内存 HMONNX 导出是后续 ModelZoo 阶段的能力。
默认关闭磁盘 offload 和 forward data parallel，量化求解器使用可见 GPU。

默认读取前 128 行，每行最多 2048 token，并显式过滤不超过 10 token 的短行。
`glm53_quantization_recipe.json` 记录请求数、实际样本数/有效 token 数、过滤数、
tokenized SHA-256、seed、格式、精度与来源。校准文本不会被写入部署包。
输出先写临时目录，完成 packed 张量清单、shape/dtype、正且有限的 scale、连续 G64
分组和 W4/W8 数量审计后原子发布。
已有输出不会覆盖；准备目录只复用来源指纹和层数匹配的完整产物。

`tests/ci/test_glm53.py` 覆盖原生/Defuser clamp、准备和参数契约；
`test_glm53_quant.py` 用保留 KDA+DSA+mHC+MoE+vision 的最小原生模型执行真实
GPTQ、保存、重载和 packed 内容检查。此合成 fixture 不代表发布权重的质量。
ModelZoo 当前 CI 的 `test_glm53_gptq.py` 另验证消费、HMONNX 和多模态生成。
