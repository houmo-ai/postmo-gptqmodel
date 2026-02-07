# r1_config_qr.py 技术文档

## 1. 脚本定位

`calibrater/r1_config_qr.py` 基于 `calibrater/r1_base_qr.py` 的核心训练逻辑开发，保留以下关键点：

- 正交旋转矩阵参数化（`QR` 投影保持正交性）
- 与原脚本一致的 Q-SR 损失主流程
- 以激活数据为输入进行旋转矩阵训练

并新增：

- JSON 配置驱动的参数管理与严格校验
- 数据集格式自动识别与适配
- 三种训练模式：`all` / `layer_wise` / `head_wise`
- 旋转矩阵独立保存（不与模型权重融合）

## 2. 完整工作流

脚本执行流程固定为：

1. 参数解析（CLI + JSON 配置）
2. 数据加载（自动识别文件组织格式）
3. 数据预处理（按模式组装样本引用）
4. 模型训练（R1 或按 head 的旋转矩阵）
5. 结果评估（Q-SR 评估损失）
6. 结果保存（标准化 `.pt` + 元信息 `.json`）

## 3. 训练模式

### 3.1 `train_data: "all"`

- 输入：目标文件夹下所有激活文件
- 输出格式：

```python
{"Q_block": R_Q}
```

其中 `R_Q` 为单个全局旋转矩阵 `torch.Tensor`。

### 3.2 `train_data: "layer_wise"`

- 输入：按层分组训练，每层独立矩阵
- 输出格式：

```python
{"Q_q_lowrank_layer_0": R0, "Q_q_lowrank_layer_1": R1, ...}
```

### 3.3 `train_data: "head_wise"`

- 输入：按层训练，并按注意力头拆分矩阵（参考 `r2_base_qr.py` 的分头旋转逻辑）
- 输出格式：

```python
{
  "Q_q_lowrank_layer_0": [R_head0, R_head1, ...],
  "Q_q_lowrank_layer_1": [R_head0, R_head1, ...]
}
```

## 4. 数据格式自动适配

脚本兼容 `calibrater/get_train_data_hook.py` 的常见产物：

- `sample_{idx}_layer_{layer_idx}_*.pt`（separate 模式）
- `layer_{layer_idx}_*.pt`（combined 模式）

并支持以下张量形状：

- `[tokens, hidden]`
- `[1, tokens, hidden]`
- `[num_samples, tokens, hidden]`

脚本会自动拆解为逻辑样本进行训练。

## 5. 输出存储规范

输出为独立旋转矩阵包，不做权重融合。保存内容包含：

- `format_version`
- `target_name` / `folder` / `train_mode`
- `rotation_matrices`
- `metrics`
- `train_args`
- `config_entry`
- `runtime_args`

同时生成同名 `.json` 元信息文件（不含大张量），用于快速检查。

## 6. 日志与进度

- 控制台 + 文件双通道日志（`--log_dir`）
- 训练中使用 `tqdm` 显示 epoch、batch、layer 进度条
- 关键节点日志：配置加载、数据扫描、训练开始/结束、评估、保存

## 7. 基本用法

### 7.1 通过 target 选择配置项

```bash
python calibrater/r1_config_qr.py \
  --model THUDM/GLM-4.7-Flash \
  --config_file config/layer_config_example.json \
  --target Q_q_lowrank \
  --calib_dataset wikitext2 \
  --calib_sample 128 \
  --nsamples 128 \
  --output_dir trained_rotation
```

### 7.2 通过 folder 选择配置项

```bash
python calibrater/r1_config_qr.py \
  --model THUDM/GLM-4.7-Flash \
  --config_file config/layer_config_example.json \
  --folder Q_block \
  --calib_dataset wikitext2 \
  --calib_sample 128
```

### 7.3 显式指定数据目录

```bash
python calibrater/r1_config_qr.py \
  --config_file config/layer_config_example.json \
  --target Q_block \
  --data_path data/wikitext2_128samples/GLM-4.7-Flash/Q_block_train
```

### 7.4 仅校验配置与数据（不训练）

```bash
python calibrater/r1_config_qr.py \
  --config_file config/layer_config_example.json \
  --target Q_block \
  --data_path data/wikitext2_128samples/GLM-4.7-Flash/Q_block_train \
  --dry_run
```
