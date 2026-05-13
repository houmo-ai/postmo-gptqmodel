# AutoRound 深度解析：架构、算法与实现

> 基于 AutoRound v0.12+ 代码库的系统学习笔记
> 适用于 Qwen3.5-27B 等大模型量化实践

---

## 目录

1. [总体架构](#1-总体架构)
2. [核心量化算法：Block-wise SignSGD](#2-核心量化算法block-wise-signsgd)
3. [校准数据加载与处理](#3-校准数据加载与处理)
4. [模型导出与格式支持](#4-模型导出与格式支持)
5. [推理加载机制](#5-推理加载机制)
6. [多模态模型处理](#6-多模态模型处理)
7. [Hadamard 变换与旋转集成](#7-hadamard-变换与旋转集成)
8. [Qwen3.5-27B 完整量化实践](#8-qwen35-27b-完整量化实践)

---

## 1. 总体架构

### 1.1 入口与工厂模式

AutoRound 使用 `__new__()` 工厂方法根据模型类型自动路由到不同的 Compressor：

```python
# auto_round/autoround.py
class AutoRound:
    def __new__(cls, model, tokenizer, **kwargs):
        # 检测模型类型 → 动态选择 Compressor
        if is_diffusion_model(model):
            return DiffusionCompressor(model, tokenizer, **kwargs)
        elif is_mllm(model):       # 多模态模型
            return MLLMCompressor(model, tokenizer, **kwargs)
        else:                       # 纯文本 LLM
            return LLMCompressor(model, tokenizer, **kwargs)
```

### 1.2 类继承层次

```
BaseCompressor (base.py)
├── LLMCompressor          # 纯文本 LLM（默认）
├── MLLMCompressor          # 多模态模型（VLM、Audio等）
├── DiffusionCompressor     # 扩散模型
├── HybridCompressor        # AR+Diffusion 混合
└── AdamCompressor          # Adam 优化器（可选mixin）
```

### 1.3 核心组件关系

```
AutoRound (入口)
  ├── BaseCompressor.quantize()        # 量化主循环
  │    ├── calib_dataset.py            # 校准数据加载
  │    ├── try_cache_inter_data()      # 激活缓存
  │    └── _quantize_blocks()          # 逐块量化
  │         ├── WrapperLinear          # 层包装器
  │         ├── SignSGD                # 优化器
  │         └── data_type/             # 量化数据类型
  ├── BaseCompressor.quantize_and_save()  # 量化+保存
  │    └── formats.py → export/        # 格式化导出
  └── special_model_handler.py         # 特殊模型处理
```

---

## 2. 核心量化算法：Block-wise SignSGD

### 2.1 算法原理

AutoRound 的核心创新是使用 **SignSGD（符号随机梯度下降）** 优化权重的取整方式。与传统 RTN（Round-To-Nearest）不同，AutoRound 通过梯度信息学习每个权重应该向上还是向下取整。

**数学形式**：
- 设原始权重 `W`，量化后权重 `Q(W)`
- 目标：最小化 `L = ||X·W - X·Q(W + V)||²`，其中 `V` 是可学习的取整偏移
- 更新规则：`V ← V - lr · sign(∂L/∂V)`（SignSGD，仅使用梯度符号）

### 2.2 SignSGD 优化器实现

```python
# auto_round/sign_sgd.py
class SGD(Optimizer):
    """SignSGD 优化器 —— 继承自 PyTorch SGD"""
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # 核心区别: 使用梯度的符号而非梯度本身
                # sign(grad) 确保每步更新幅度固定为 lr
                p.add_(d_p.sign(), alpha=-group['lr'])
```

**为什么用 SignSGD？**
- 量化取整是离散优化，梯度大小意义不大，方向（sign）才重要
- 固定步长避免学习率调参
- 计算高效，内存占用小

### 2.3 WrapperLinear 层包装器

WrapperLinear 是 AutoRound 的核心组件，它包装每个需要量化的 Linear 层，添加可学习参数：

```python
# auto_round/wrapper.py
class WrapperLinear(nn.Module):
    def __init__(self, orig_layer, ...):
        # 可学习参数
        self.weight_value = nn.Parameter(torch.zeros_like(weight))  # 取整偏移 V
        self.min_scale = nn.Parameter(torch.ones(...))              # 量化范围最小缩放
        self.max_scale = nn.Parameter(torch.ones(...))              # 量化范围最大缩放
        
    def forward(self, x):
        # 1. 应用缩放调整量化范围
        adjusted_weight = self.weight + self.weight_value
        
        # 2. 执行量化 (fake quantize)
        q_weight = quant_weight(adjusted_weight, 
                                bits=self.bits,
                                group_size=self.group_size,
                                min_scale=self.min_scale,
                                max_scale=self.max_scale)
        
        # 3. 计算输出 (STE: Straight-Through Estimator)
        output = F.linear(x, q_weight, self.bias)
        return output
```

**Straight-Through Estimator (STE)**：量化操作不可微，但通过 STE，梯度直接穿过量化函数传到连续参数 `weight_value`。

### 2.4 Block-wise 量化循环

```python
# auto_round/autoround.py - 简化版量化主循环
def quantize(self):
    # Step 1: 缓存块间激活
    cached_inputs = self.try_cache_inter_data_gpucpu()
    
    # Step 2: 逐块量化
    for block_idx, block in enumerate(self.blocks):
        self._quantize_blocks(block, cached_inputs[block_idx])

def _quantize_blocks(self, block, block_inputs):
    # 1. 包装: 将 block 内所有 Linear 替换为 WrapperLinear
    wrapped_layers = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear) and name not in ignore_list:
            wrapped = WrapperLinear(module, bits, group_size, ...)
            wrapped_layers[name] = wrapped
    
    # 2. 优化循环
    if self.iters > 0:
        optimizer = SignSGD(all_params, lr=self.lr)
        
        for iteration in range(self.iters):
            for batch in block_inputs:
                # Forward pass → 计算量化误差
                output_q = block(batch)
                output_fp = block_fp(batch)  # FP 参考输出
                loss = mse_loss(output_q, output_fp)
                
                # Backward → SignSGD update
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录最优参数
            if loss < best_loss:
                save_best_params()
    
    # 3. 应用最优量化参数并解包
    for name, wrapper in wrapped_layers.items():
        apply_quantized_weight(wrapper)
        unwrap(block, name, wrapper)
```

### 2.5 量化数据类型

AutoRound 支持多种数据类型（`auto_round/data_type/`）：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `int` | 均匀整数量化 | 通用 4/8-bit |
| `nf4` | NormalFloat 4-bit | QLoRA 兼容 |
| `fp4_e2m1` | FP4 (2 exp, 1 mantissa) | 低精度推理 |
| `fp8_e4m3` | FP8 (4 exp, 3 mantissa) | 高精度推理 |
| `mx_fp4/mx_fp8` | Microscaling FP | 硬件加速 |

---

## 3. 校准数据加载与处理

### 3.1 数据集注册机制

```python
# auto_round/calib_dataset.py
CALIB_DATASETS = {}  # 全局注册表

def register_dataset(name):
    """装饰器注册数据集"""
    def decorator(func):
        if isinstance(name, list):
            for n in name:
                CALIB_DATASETS[n] = func
        else:
            CALIB_DATASETS[name] = func
        return func
    return decorator

# 使用示例
@register_dataset(["NeelNanda/pile-10k", "pile-10k"])
def get_pile_dataset(tokenizer, seqlen, dataset_name, ...):
    dataset = load_dataset(dataset_name, split="train")
    # ... tokenize and return
```

### 3.2 支持的数据集

| 数据集 | 类型 | 特点 |
|--------|------|------|
| `NeelNanda/pile-10k` | 通用文本 | **默认推荐**，对 AutoRound 效果最好 |
| `swift/pile-val-backup` | 通用文本 | ModelScope 备份（解决网络问题） |
| `BAAI/CCI3-HQ` | 指令数据 | 中文优化 |
| `codeparrot/github-code-clean` | 代码 | 代码生成优化 |
| 自定义 DataLoader | 任意 | 用户提供 |

### 3.3 数据处理流水线

```
原始数据集
  ↓ load_dataset() / 用户DataLoader
文本数据
  ↓ get_tokenizer_function()
  ↓ [可选] apply_chat_template() → 使用对话模板格式化
  ↓ tokenize → padding/truncation to seqlen (默认 2048)
Token IDs
  ↓ DataLoader(batch_size, shuffle)
校准 Batches
  ↓ try_cache_inter_data_gpucpu()
  ↓ 模型前向传播 → 缓存每个 block 边界的激活值
Cached Activations (per block)
  → 供 _quantize_blocks() 使用
```

### 3.4 关键发现

**自生成数据对 AutoRound 有害**：
- pile-10k: PPL +2.07%
- self-generated: PPL +2.36%
- 原因：自生成数据是为 GPTQ 的 Hessian-MSE 策略优化的，不适合 SignSGD
- **结论：AutoRound 应始终使用通用数据集 pile-10k**

---

## 4. 模型导出与格式支持

### 4.1 格式注册系统

```python
# auto_round/formats.py
class OutputFormat:
    """导出格式基类"""
    FORMATS = {}  # 全局格式注册表
    
    @classmethod
    def register(cls, name, format_class):
        cls.FORMATS[name] = format_class

# 8种格式注册
OutputFormat.register("auto_round", AutoRoundFormat)
OutputFormat.register("fake", FakeFormat)
OutputFormat.register("gguf", GGUFFormat)
OutputFormat.register("gptq", AutoGPTQFormat)
OutputFormat.register("gptqmodel", GPTQModelFormat)
OutputFormat.register("awq", AutoAWQFormat)
OutputFormat.register("llmcompressor", LLMCompressorFormat)
OutputFormat.register("fp8", FP8Format)
```

### 4.2 导出格式对比

| 格式 | 后端 | 打包方式 | 适用场景 |
|------|------|---------|---------|
| **auto_round** | auto_round_extension | safetensors + qweight打包 | vLLM 部署，推荐 |
| **fake** | 无内核 | 仅保存反量化权重 | 精度评估 |
| **gguf** | llama.cpp | GGUF 二进制 | 边缘设备 (ollama等) |
| **gptq** | AutoGPTQ | safetensors + g_idx | 兼容 AutoGPTQ 生态 |
| **gptqmodel** | GPTQModel | safetensors | GPTQModel 生态 |
| **awq** | AutoAWQ | safetensors | AWQ 推理 |
| **llmcompressor** | llmcompressor | 元数据格式 | LLM Compressor 生态 |
| **fp8** | FP8 内核 | per-channel scaling | 高精度 FP8 推理 |

### 4.3 保存流程

```python
# auto_round/autoround.py - quantize_and_save()
def quantize_and_save(self, output_dir, format="auto_round,fake", inplace=True):
    # 1. 执行量化
    model, quantize_config = self.quantize()
    
    # 2. 解析格式列表（支持逗号分隔多格式）
    formats = format.split(",")
    
    # 3. 逐格式保存
    folders = []
    for fmt in formats:
        fmt_class = OutputFormat.FORMATS[fmt.strip()]
        folder = fmt_class.save_quantized(
            model=model,
            output_dir=f"{output_dir}_{fmt}" if len(formats) > 1 else output_dir,
            quantize_config=quantize_config,
            tokenizer=self.tokenizer,
        )
        folders.append(folder)
    
    return model, folders
```

### 4.4 保存的文件结构

```
output_dir/
├── config.json                    # 模型配置 + 量化元数据
│   └── quantization_config: {
│         "bits": 4,
│         "group_size": 128,
│         "sym": true,
│         "data_type": "int",
│         "iters": 200,
│         "nsamples": 128,
│         "autoround_version": "0.12.0"
│       }
├── generation_config.json
├── model.safetensors              # 量化后的模型权重
│   └── 包含: qweight (packed), scales, zeros
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

---

## 5. 推理加载机制

### 5.1 加载流程

```python
# 方式1: 使用 AutoRound 加载
from auto_round import AutoRoundConfig
model = AutoModelForCausalLM.from_pretrained(
    "quantized_model_path",
    quantization_config=AutoRoundConfig()
)

# 方式2: transformers 自动检测
# 如果 config.json 中包含 quantization_config，transformers 会自动路由
model = AutoModelForCausalLM.from_pretrained("quantized_model_path")
```

### 5.2 反量化过程

加载时，量化权重通过对应后端的内核进行在线反量化：

```
加载 safetensors → 读取 qweight (packed int4) + scales + zeros
  ↓
推理时: dequant(qweight, scales, zeros) → FP16 weight → matmul(input, weight)
  ↓
或: 使用 fused kernel 直接在 int4 上计算 (更快)
```

---

## 6. 多模态模型处理

### 6.1 MLLM 检测与路由

```python
# auto_round/special_model_handler.py
# 已知的多模态模型列表
mllms_with_limited_bs = {
    "llava", "qwen2_vl", "qwen2_5_vl", "phi3_v", "mllama",
    "qwen2_5_omni", "qwen3_omni_moe", "glm_image"
}

# 支持纯文本校准的多模态模型
SUPPORT_ONLY_TEXT_MODELS = [
    "llava", "qwen2_vl", "qwen2_5_vl", "deepseek_vl_v2",
    "qwen2_5_omni", "qwen3_omni_moe", "glm_image", ...
]
```

### 6.2 多模态模型量化策略

```
完整 VLM 模型
├── 视觉编码器 (Vision Encoder)  → 默认不量化，保留全精度
├── 视觉适配器 (Merger/Adapter)  → 默认不量化
└── 文本解码器 (LLM Backbone)    → 量化目标
    └── transformer.layers[0:N]  → Block-wise 量化
```

### 6.3 特殊多模态块提取

```python
SPECIAL_MULTIMODAL_BLOCK = {
    "deepseek_vl_v2": _get_deepseek_vl2_multimodal_block,
    "qwen2_5_omni": _get_qwen2_5_omni_multimodal_block,
    "qwen3_omni_moe": _get_qwen3_omni_moe_multimodal_block,
    "glm_image": _get_glm_image_multimodal_block,
}

# 示例: Qwen2.5-Omni
def _get_qwen2_5_omni_multimodal_block(model):
    return {
        "text_blocks": model.thinker.model.layers,      # 量化这些
        "vision_blocks": model.thinker.visual.blocks,     # 可选量化
        "audio_blocks": model.thinker.audio_tower,        # 可选量化
    }
```

### 6.4 Qwen3.5 的特殊情况

**问题**: Qwen3.5 是 `ForConditionalGeneration`（VLM 架构），但当前实验中强制作为 `CausalLM` 加载。

**影响**:
- 丢失视觉编码器权重
- 量化后模型不完整
- 无法进行视觉推理

**正确做法**: 使用 MLLMCompressor 路径，或在保存时确保视觉权重完整保留。

---

## 7. Hadamard 变换与旋转集成

### 7.1 两种变换方式

| | 内置变换 | 外部旋转 |
|--|---------|---------|
| **位置** | `experimental/transform/` | 外部脚本 (GPTQModel) |
| **时机** | 量化过程中 online 应用 | 量化前 offline 预处理 |
| **范围** | 逐层/逐组 | 全模型 |
| **目标** | MXFP4/MXFP8 格式优化 | 权重/激活分布平滑 |
| **可逆性** | 运行时逆变换 | 永久修改权重 |

### 7.2 内置 Hadamard 变换

```python
# auto_round/experimental/transform/transforms.py
class HadamardTransform:
    """确定性 Hadamard 变换（Sylvester 构造）"""
    def __init__(self, block_size=32):
        self.H = deterministic_hadamard_matrix(block_size)
        self.scale = 1.0 / math.sqrt(block_size)
    
    def forward(self, x):
        return x @ self.H * self.scale
    
    def inverse(self, x):
        return x @ self.H.T * self.scale  # Hadamard 矩阵是正交的: H.T = H^{-1}
```

### 7.3 外部旋转（GPTQModel 方式）

```python
# 旋转流程 (rotate_and_quantize.py 使用的方式)
from gptqmodel.quantization.rotation import rotation

# 1. 融合 LayerNorm 到 Linear
fuse_layer_norms(model)  # W_new = W * norm_weight

# 2. 生成正交旋转矩阵
Q = random_hadamard_matrix(hidden_size, device)  # Q·Q^T = I

# 3. 旋转所有权重
# Embedding:  W_emb = W_emb @ Q
# Q/K/V proj: W_qkv = W_qkv @ Q
# O proj:     W_o = Q^T @ W_o
# Gate/Up:    W_gu = W_gu @ Q
# Down:       W_down = Q^T @ W_down
# LM Head:    W_head = Q^T @ W_head

# 4. (可选) 旋转视觉 Merger
# Merger FC2: W_fc2 = Q^T @ W_fc2
```

### 7.4 视觉旋转（GPTQModel 参考实现）

GPTQModel 对视觉模块的旋转分为两个级别：

```python
# 级别1: merger_only（必须，当 LLM 旋转时）
def rotate_visual_merger_qwen3_5_vl(model, Q, device):
    """旋转 merger 输出层以匹配旋转后的 LLM 输入空间"""
    merger = model.visual.merger
    # W_fc2 = Q^T @ W_fc2
    w = merger.linear_fc2.weight.data.to(device=device, dtype=torch.float64)
    merger.linear_fc2.weight.data = (Q.T @ w).to(original_dtype)
    if merger.linear_fc2.bias is not None:
        b = merger.linear_fc2.bias.data.to(device=device, dtype=torch.float64)
        merger.linear_fc2.bias.data = (Q.T @ b).to(original_dtype)

# 级别2: full（可选，旋转完整视觉编码器）
def rotate_vision_model_qwen3_5_vl(model, Q_vision, device):
    """旋转完整视觉编码器"""
    visual = model.visual
    # 旋转 patch embedding
    rotate_conv(visual.patch_embed.proj, Q_vision, ...)
    # 旋转所有 vision transformer blocks
    for block in visual.blocks:
        rotate_attention(block.attn, Q_vision)
        rotate_mlp(block.mlp, Q_vision)
    # 旋转 merger 输入
    rotate_merger_input(visual.merger, Q_vision)
```

**数学等价性**:
```
旋转前: output = LLM(Merger(Vision(image)))
旋转后: output = LLM'(Merger'(Vision(image)))  [merger_only]
       = LLM'(Merger'(Vision'(image)))          [full rotation]
其中 LLM' 和 Merger' 保证输出等价
```

---

## 8. Qwen3.5-27B 完整量化实践

### 8.1 模型概览

| 参数 | 值 |
|------|-----|
| 架构 | Qwen3.5ForConditionalGeneration (VLM) |
| 参数量 | 27B |
| Layers | 64 |
| Hidden Size | 5120 |
| 视觉编码器 | 有 (Qwen3.5 VL) |
| 模型类型 | 多模态 (文本 + 视觉) |

### 8.2 纯 AutoRound 量化

```bash
# quantize_autoround.py - 纯 AutoRound W4G64
python quantize_autoround.py \
    --model_name Qwen/Qwen3.5-27B \
    --bits 4 \
    --group_size 64 \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --dataset "NeelNanda/pile-10k" \
    --format "auto_round" \
    --output_dir ./qwen35-27b-w4g64-autoround
```

**注意事项**:
- 当前脚本使用 `AutoModelForCausalLM` 加载以避免 MLLM 检测，会丢失视觉部分
- 建议改用完整 VLM 加载路径（见 8.4）

### 8.3 Rotate + AutoRound 量化

```bash
# rotate_and_quantize.py - 旋转 + AutoRound
python rotate_and_quantize.py \
    --model_name Qwen/Qwen3.5-27B \
    --bits 4 \
    --group_size 64 \
    --iters 200 \
    --rotation_mode hadamard \
    --output_dir ./qwen35-27b-w4g64-rotated
```

**流程**:
1. 加载完整模型
2. 融合 RMSNorm → Linear
3. 应用 Hadamard 旋转（~232s）
4. AutoRound 量化（~8521s ≈ 2.4h）
5. 保存量化模型

### 8.4 推荐的完整 VLM 量化流程（待实现）

```python
# 推荐: 保留视觉部分的完整量化
from auto_round import AutoRound
from transformers import AutoProcessor

# 1. 以 VLM 方式加载完整模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-27B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")

# 2. (可选) 应用旋转 - 包括视觉 merger
apply_rotation(model, mode="hadamard", rotate_vision="merger_only")

# 3. AutoRound 量化 (自动检测为 MLLM)
ar = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=4,
    group_size=64,
    iters=200,
    dataset="NeelNanda/pile-10k",
)

# 4. 量化并保存（包含完整视觉权重）
model, folders = ar.quantize_and_save(
    output_dir="./qwen35-27b-w4g64-complete",
    format="auto_round"
)
```

### 8.5 实验结果对比

| 方案 | PPL (wikitext-2) | PPL 增幅 | 对话质量 | 激活分布 |
|------|:---:|:---:|:---:|:---:|
| BF16 原始 | 7.5335 | — | 基准 | 原始 |
| AutoRound W4G64 (pile-10k) | 7.6891 | **+2.07%** | 无损 | 原始 |
| Rotate+AutoRound W4G64 (pile-10k) | 7.6937 | +2.13% | 无损 | **更均匀** |
| AutoRound W4G64 (self-gen) | 7.7113 | +2.36% | 无损 | 原始 |
| Rotate+RTN W4G64 | 7.7829 | +3.31% | 无损 | 更均匀 |

**结论**: 
- AutoRound + Rotate 为**生产优选**（激活量化友好，芯片部署性能更好）
- PPL 差异仅 0.06%，可忽略
- **坚持使用 pile-10k 数据集**，不要使用自生成数据

---

## 附录 A: 关键配置参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bits` | 4 | 量化位宽 |
| `group_size` | 128 | 量化分组大小 |
| `sym` | False | 对称/非对称量化 |
| `iters` | 200 | SignSGD 迭代次数（0=RTN） |
| `lr` | 1.0/iters | 学习率 |
| `nsamples` | 128 | 校准样本数 |
| `seqlen` | 2048 | 序列长度 |
| `batch_size` | 8 | 校准批大小 |
| `dataset` | "NeelNanda/pile-10k" | 校准数据集 |
| `format` | "auto_round" | 导出格式 |
| `enable_minmax_tuning` | True | 量化范围微调 |
| `low_gpu_mem_usage` | False | 低显存模式 |
| `amp` | True | 混合精度训练 |

## 附录 B: 文件索引

| 文件 | 职责 |
|------|------|
| `auto_round/autoround.py` | 入口 + 工厂 + 量化主循环 |
| `auto_round/wrapper.py` | WrapperLinear 层包装器 |
| `auto_round/sign_sgd.py` | SignSGD 优化器 |
| `auto_round/calib_dataset.py` | 校准数据集加载 |
| `auto_round/formats.py` | 导出格式注册 |
| `auto_round/schemes.py` | 量化方案定义 |
| `auto_round/special_model_handler.py` | 特殊模型处理 |
| `auto_round/export/` | 格式化导出实现 |
| `auto_round/inference/` | 推理加载 |
| `auto_round/data_type/` | 量化数据类型 |
| `auto_round/experimental/transform/` | Hadamard 变换 |
| `auto_round/compressors/mllm/` | 多模态压缩器 |
