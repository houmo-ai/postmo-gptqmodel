# XH2a的GPTQ

## 1. 安装依赖

```bash
#setuptools 80之后的版本采用 /opt/extdata/.conda/envs/torch2_7/bin/python -m pip install -e . --use-pep517 --no-build-isolation --config-settings editable_mode=compat ， 多了--use-pep517参数会导致找不到torch
pip install setuptools==79.0.1
pip uninstall SoundFile
pip install SoundFile
```

## 2. 自研算法加持

```bash
1. 支持自研算法**MoeQuant**：用于moe模型校准数据集的生成，使得moe专家都能获得合适的校准数据去做gptq，example在gptqmodel/quantization/ebss
2. 支持自研算法**dicequant**：搜集对应位置激活，激活进行训练获取旋转矩阵，算法实现在gptqmodel/quantization/diceq，example在gptqmodel/models/definitions/glm4_moe_lite.py中的diceq_config_template里的配置
3. 支持自研算法**混合精度**：针对moe模型，根据模型结构和激活分布，自动搜索最优的混合精度配置，example在gptqmodel/models/definitions/glm4_moe_lite.py中的mix_precision_config_template里的配置
```

## 3. 特定模型需要特定版本的仓库

```bash
deepseekocrv1结构： transformers==4.46.3
deepseekv2/v3结构/glm4_moe_lite： transformers==5.0.0rc3
```

## 4. 算法流程图

```Mermaid
graph LR
    %% 定义样式
    classDef main fill:#4A90E2,stroke:#357ABD,color:white,font-weight:bold;
    classDef unfinished fill:#FADBD8,stroke:#E74C3C,color:#7B241C;
    classDef risk fill:#FFF9C4,stroke:#FBC02D,color:#7F8C8D;

    %% --- Pipeline 横向流程 ---
    subgraph Pipeline_Flow [Pipeline 流水线]
    direction LR
    P1[pipeline]:::main --> P2(校准集准备):::unfinished
    P2 --> P3(模型convert)
    P3 --> P4(fuse_norm)
    P4 --> P5(diceq)
    P5 --> P6(rotation)
    P6 --> P7(混合精度):::unfinished
    P7 --> P8(gptq)
    P8 --> P9(输出量化模型+每层量化config)
    end
```

```Mermaid
graph LR
    %% 定义样式
    classDef main fill:#4A90E2,stroke:#357ABD,color:white,font-weight:bold;
    classDef unfinished fill:#FADBD8,stroke:#E74C3C,color:#7B241C;
    classDef risk fill:#FFF9C4,stroke:#FBC02D,color:#7F8C8D;

    %% --- gptqmodel 逻辑树 ---
    Root[gptqmodel]:::main

    %% 激活分支
    Root --> Activation[激活]
    Activation --- A8[A8/A16]
    A8 --- RTN1[RTN: 针对一些cv模型]

    Activation --- A5[A5/A4]
    A5 --> A_Easy[易量化]
    A_Easy --- Quaro1[quarot: 易量化LLM模型]
    A_Easy --- BR1(Block rotation)

    A5 --> A_Hard[难量化]
    A_Hard --- Diceq1(diceq: 针对激活训练)
    A_Hard --- Mixed1(混合精度 v1: 基于quarot/diceq<br/>只针对moe/mlp合并的暴力搜索):::unfinished
    Mixed1 --- SQ1[singlequant: 只做克罗克小矩阵的16bit matmul]:::unfinished
    Mixed1 --- Bit1[提升bit, 导出config给xhmodelzoo]:::unfinished

    %% 权重分支
    Root --> Weights[权重]
    Weights --- W8[w8]
    W8 --- RTN2[RTN: 针对sLLM, cv模型]

    Weights --- W5[w5/w4]
    W5 --> W_Easy[易量化]
    W_Easy --- GPTQ2[gptq: 针对权重训练]
    GPTQ2 --- Mixed3(需增加根据混合精度搜索的混合精度的gptq):::unfinished
    W_Easy --- Hessian[hessian mse]
    W_Easy --- Rot2[rotation]
    Rot2 --- Quaro2[quarot: 易量化LLM模型]
    Rot2 --- BR2(Block rotation)

    W5 --> W_Hard[难量化]
    W_Hard --- Diceq2(diceq: 针对权重训练)
    W_Hard --- Mixed2(混合精度 v1: 基于quarot/diceq<br/>只针对moe/mlp合并的暴力搜索):::unfinished
    Mixed2 --- Lora[lora: linear附加lora补偿gptq残差]
    Mixed2 --- Bit2[提升bit做gptq]

    W_Hard --> Calib[校准集]
    Calib --- MoeQ(moe: moequant需要pileline到整个流程):::unfinished
    Calib --- DenseQ(dense: calibration v2):::unfinished

    %% 风险点
    Root --> Risks[剩余风险点]
    Risks --- R1(moe算子内部down_proj输入无法优化可smooth):::risk
    Risks --- R2(需要替换gptq的quantizer为sefp与ssfp方式):::risk
```

## 5. TODO List

- [ ] diceq算法：
  - [ ] 对于ov_rotate,需要支持head_wise的旋转，这块目前还没支持，还没验证
- [ ] 校准集生成：
  - [ ] ebss集成到pipeline中，moe模型默认开启
  - [ ] ivsg集成到pipeline中，默认不开启
- [ ] 混合精度：
  - [ ] moe/mlp 算子当成一个整体去搜索，给出config
  - [ ] 激活混合精度
    - [ ] 配置1：将singlequant这种小克罗内克旋转矩阵
    - [ ] 配置2：提升bit
  - [ ] 权重混合精度
    - [ ] 配置1：lora去补偿权重损失
    - [ ] 配置2：提升bit，导出config给gptq
- [ ] 替换gptq的quantizer为sefp/ssfp的
- [ ] moe算子内部的down_proj应用smooth平滑优化
