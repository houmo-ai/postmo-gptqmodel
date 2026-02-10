# XH2a的GPTQ

## 安装依赖

1. 环境配置

```bash
#setuptools 80之后的版本采用 /opt/extdata/.conda/envs/torch2_7/bin/python -m pip install -e . --use-pep517， 多了--use-pep517参数会导致找不到torch
pip install setuptools==79.0.1
pip uninstall SoundFile
pip install SoundFile
```

2. 自研算法加持

```bash
1. 支持自研算法**MoeQuant**：用于moe模型校准数据集的生成，使得moe专家都能获得合适的校准数据去做gptq，example在gptqmodel/quantization/ebss
2. 支持自研算法**dicequant**：搜集对应位置激活，激活进行训练获取旋转矩阵，算法实现在gptqmodel/quantization/diceq，example在gptqmodel/models/definitions/glm4_moe_lite.py中的diceq_config_template里的配置
```

3. 特定模型需要特定版本的仓库
```bash
deepseekv2/deepseekocrv1结构： transformers==4.46.3
deepseekv3结构/glm4_moe_lite： transformers==5.0.0rc3
```