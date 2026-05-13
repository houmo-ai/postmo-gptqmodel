# Qwen3.5-27B 量化前后 Think 能力对比报告

## 实验设置

| 模型 | 路径 | 量化方式 |
|------|------|----------|
| HF (BF16) | weights/Qwen3.5-27B | 无 (原始) |
| GPTQ | weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse | W4G64, gptqmodel, hessian+mse |
| AutoRound | weights/Qwen3.5-27B-W4G64-autoround/Qwen3.5-27B-w4g64 | W4G64, auto-round |
| AutoRound+Rotate | weights/Qwen3.5-27B-W4G64-rotate-autoround/Qwen3.5-27B-rotated-w4g64 | W4G64, rotate+auto-round |

- `enable_thinking=True`, `do_sample=False`, `max_new_tokens=2048`
- 5 个问题，GPU: A100-80G

## 1. 汇总统计 (5 题合计)

| 模型 | Think Tokens | Answer Tokens | Think占比 | Think/Answer比 |
|------|-------------|---------------|----------|---------------|
| HF | 1863 | 3872 | 32.5% | 0.48 |
| GPTQ | 2711 | 3541 | 43.4% | 0.77 |
| AutoRound | 1619 | 3852 | 29.6% | 0.42 |
| AutoRound+Rotate | 1987 | 3596 | 35.6% | 0.55 |

## 2. 逐题 Think Token 数对比

| 问题 | HF | GPTQ | AutoRound | AutoRound+Rotate | 备注 |
|------|-----|------|-----------|------------------|------|
| 你是谁？ | 309 | 856 | 258 | 249 |  |
| 中国最长的河流是什么？请简要介绍。 | 615 | 916 | 619 | 1045 |  |
| 请解方程 2x + 5 = 15 | 813 | 812 | 590 | 572 |  |
| 用100字描述春天的景色 | 0 | 0 | 0 | 0 | ⚠ 思考未完成(达到2048上限) |
| 写一个Python函数计算斐波那契数列的第n项 | 126 | 127 | 152 | 121 |  |

## 3. 逐题 Answer Token 数对比

| 问题 | HF | GPTQ | AutoRound | AutoRound+Rotate |
|------|-----|------|-----------|------------------|
| 你是谁？ | 217 | 39 | 246 | 73 |
| 中国最长的河流是什么？请简要介绍。 | 134 | 152 | 123 | 158 |
| 请解方程 2x + 5 = 15 | 207 | 125 | 131 | 142 |
| 用100字描述春天的景色 | 2050 | 2048 | 2048 | 2049 |
| 写一个Python函数计算斐波那契数列的第n项 | 1264 | 1177 | 1304 | 1174 |

## 4. Think 啰嗦度分析 (排除Q4 - 春天景色)

| 模型 | Think Tokens (4题) | Answer Tokens (4题) | Think/Answer | 相对HF变化 |
|------|-------------------|--------------------|--------------|-----------| 
| HF | 1863 | 1822 | 1.02 | — |
| GPTQ | 2711 | 1493 | 1.82 | +45.5% |
| AutoRound | 1619 | 1804 | 0.90 | -13.1% |
| AutoRound+Rotate | 1987 | 1547 | 1.28 | +6.7% |

## 5. Think 内容对比 (摘录)

### Q1: 你是谁？

**HF** (think: 309 tok, answer: 217 tok)
> 嗯，用户问“你是谁？”，这是一个非常基础的自我介绍问题。作为 Qwen3.5，我需要清晰、简洁地说明自己的身份，同时突出关键能力，避免信息过载。  首先，我应该明确说明我是通义千问系列中最新推出的 Qwen3.5，由通义实验室研发。然后，需要简要概括核心升级点，比如上下文窗口、多语言支持、逻辑推理等，但不用展开技术细节。  用户可能想了解我的能力范围，是否适合他们的需求。所以应该提到主要应用场景，比如处理长文档、多语言任务、代码生成等。同时，保持友好，邀请用户提出具体任务，这样能引导对话继续。  要注意避免使用过于专业的术语，保持口语化。比如“上下文窗口”可以简化为“能处理超长文档”，“逻辑推...

**GPTQ** (think: 856 tok, answer: 39 tok)
> Thinking Process:  1.  **Analyze the Request:**     *   Input: "你是谁？" (Who are you?)     *   Language: Chinese.     *   Intent: The user is asking about my identity.  2.  **Determine My Identity:**     *   I am Qwen3.5, a large language model developed by Alibaba Cloud.     *   I need to introduce m...

**AutoRound** (think: 258 tok, answer: 246 tok)
> 嗯，用户问“你是谁？”，这是一个很基础的自我介绍问题。我需要先明确自己的身份。我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型。不过用户可能更关心我的能力和用途，而不仅仅是版本号。  首先，我应该简洁地说明我是通义千问系列的最新版本，然后突出关键能力，比如多语言支持、长上下文处理、逻辑推理等。但要注意不要堆砌技术术语，用用户能理解的语言。  用户可能想了解我能做什么，比如回答问题、写代码、分析文档等。需要分点说明，但保持口语化。另外，要提到我的训练数据截止时间是 2026 年，这样用户知道我的知识范围。  还要注意避免过于冗长，用户可能只需要基本信息。是否需要询问用户具体需求？比如“...

**AutoRound+Rotate** (think: 249 tok, answer: 73 tok)
> 嗯，用户问“你是谁？”，这是一个很基础的自我介绍问题。我需要明确自己的身份，同时保持友好和简洁。  首先，我应该说明我是通义千问，是通义实验室研发的超大规模语言模型。然后，可以提到我的版本是 Qwen3.5，但可能不需要太详细的技术参数，避免让用户觉得复杂。  接下来，要突出我的能力，比如回答问题、创作文字、编程、逻辑推理等。但要注意不要罗列太多功能，保持自然。用户可能想了解我的用途，所以可以简单举例，比如写故事、写公文、写邮件等。  还要注意语气亲切，用口语化的表达，避免太正式。比如用“你好呀！”开头，显得友好。同时，要邀请用户提问，促进互动。  需要检查是否有错误信息，比如版本号是否正确，...

### Q2: 中国最长的河流是什么？请简要介绍。

**HF** (think: 615 tok, answer: 134 tok)
> Thinking Process:  1.  **Analyze the Request:**     *   Question: "中国最长的河流是什么？" (What is the longest river in China?)     *   Requirement: "请简要介绍。" (Please introduce briefly.)     *   Language: Chinese.  2.  **Identify the Core Fact:**     *   The longest river in China is the Yangtze River (长江 - Ch...

**GPTQ** (think: 916 tok, answer: 152 tok)
> Thinking Process:  1.  **Analyze the Request:**     *   Question: "中国最长的河流是什么？" (What is the longest river in China?)     *   Requirement: "请简要介绍。" (Please briefly introduce it.)     *   Language: Chinese.  2.  **Identify the Answer:**     *   The longest river in China is the Yangtze River (长江 - Ch...

**AutoRound** (think: 619 tok, answer: 123 tok)
> Thinking Process:  1.  **Analyze the Request:**     *   Question: "中国最长的河流是什么？" (What is the longest river in China?)     *   Requirement: "请简要介绍。" (Please introduce briefly.)     *   Language: Chinese.  2.  **Identify the Core Fact:**     *   The longest river in China is the Yangtze River (长江 - Ch...

**AutoRound+Rotate** (think: 1045 tok, answer: 158 tok)
> Thinking Process:  1.  **Analyze the Request:**     *   Question: "中国最长的河流是什么？" (What is the longest river in China?)     *   Requirement: "请简要介绍。" (Please introduce briefly.)     *   Language: Chinese.  2.  **Identify the Fact:**     *   The longest river in China is the Yangtze River (长江 - Chángji...

### Q3: 请解方程 2x + 5 = 15

**HF** (think: 813 tok, answer: 207 tok)
> Here's a thinking process that leads to the suggested solution:  1.  **Analyze the Request:**     *   **Input:** "请解方程 2x + 5 = 15" (Please solve the equation 2x + 5 = 15).     *   **Goal:** Find the value of $x$.     *   **Language:** Chinese.  2.  **Step-by-Step Solution Strategy:**     *   **Equa...

**GPTQ** (think: 812 tok, answer: 125 tok)
> Here's a thinking process that leads to the solution of the equation $2x + 5 = 15$:  1.  **Analyze the Request:**     *   **Input:** "请解方程 2x + 5 = 15" (Please solve the equation 2x + 5 = 15).     *   **Goal:** Find the value of $x$.     *   **Language:** Chinese.  2.  **Identify the Equation:**    ...

**AutoRound** (think: 590 tok, answer: 131 tok)
> Here's a thinking process that leads to the solution:  1.  **Analyze the Request:**     *   The user wants to solve a linear equation: $2x + 5 = 15$.     *   The language is Chinese.  2.  **Identify the Goal:**     *   Find the value of $x$.  3.  **Step-by-Step Solution:**     *   *Equation:* $2x + ...

**AutoRound+Rotate** (think: 572 tok, answer: 142 tok)
> Thinking Process:  1.  **Analyze the Request:** The user wants me to solve the linear equation $2x + 5 = 15$.  2.  **Identify the Equation:** $2x + 5 = 15$.  3.  **Determine the Goal:** Find the value of $x$.  4.  **Step-by-Step Solution:**     *   **Step 1:** Isolate the term with $x$. To do this, ...

### Q4: 用100字描述春天的景色

**HF** (think: 0 tok, answer: 2050 tok)
> 未产生 think 或思考未完成

**GPTQ** (think: 0 tok, answer: 2048 tok)
> 未产生 think 或思考未完成

**AutoRound** (think: 0 tok, answer: 2048 tok)
> 未产生 think 或思考未完成

**AutoRound+Rotate** (think: 0 tok, answer: 2049 tok)
> 未产生 think 或思考未完成

### Q5: 写一个Python函数计算斐波那契数列的第n项

**HF** (think: 126 tok, answer: 1264 tok)
> 用户想要一个计算斐波那契数列第n项的Python函数。斐波那契数列是：0, 1, 1, 2, 3, 5, 8, 13, 21...  我应该提供几种不同的实现方法，从简单到优化，让用户了解不同的方案：  1. 递归方法（简单但效率低） 2. 迭代方法（效率高） 3. 带缓存的递归方法（动态规划） 4. 矩阵快速幂方法（最优）  我会重点推荐迭代方法，因为它简单且效率高。...

**GPTQ** (think: 127 tok, answer: 1177 tok)
> 用户想要一个计算斐波那契数列第n项的Python函数。斐波那契数列是：0, 1, 1, 2, 3, 5, 8, 13, 21...  我应该提供几种不同的实现方法，从简单到优化的版本： 1. 递归方法（简单但效率低） 2. 迭代方法（效率高） 3. 记忆化递归（折中方案） 4. 矩阵快速幂（最优但复杂）  我会重点推荐迭代方法，因为它在时间和空间效率上都是最优的。...

**AutoRound** (think: 152 tok, answer: 1304 tok)
> 用户想要一个计算斐波那契数列第n项的Python函数。我应该提供几种不同的实现方法，从简单到优化，并说明它们的优缺点。  斐波那契数列：0, 1, 1, 2, 3, 5, 8, 13, 21... F(0) = 0 F(1) = 1 F(n) = F(n-1) + F(n-2)  我会提供： 1. 递归方法（简单但效率低） 2. 迭代方法（推荐） 3. 带缓存的递归方法 4. 矩阵快速幂方法（高级）  让我写一个完整的示例。...

**AutoRound+Rotate** (think: 121 tok, answer: 1174 tok)
> 用户想要一个计算斐波那契数列第n项的Python函数。斐波那契数列是：0, 1, 1, 2, 3, 5, 8, 13, 21...  我应该提供几种不同的实现方法，从简单到优化，并说明各自的优缺点。  1. 递归方法（简单但效率低） 2. 迭代方法（效率高） 3. 记忆化递归（平衡） 4. 矩阵快速幂（最优，适合大n）  让我写一个完整的解决方案。...

## 6. 结论

1. **GPTQ 量化使 think 明显变啰嗦**: think token 比 HF **增加 +45.5%**，think/answer 比从 1.02 升至 1.82，尤其 Q1(你是谁) GPTQ think 856 tok vs HF 309 tok (+177%)
2. **AutoRound 量化 think 略微减少**: think token 比 HF **减少 -13.1%**，think/answer 比 0.90，与 HF 接近
3. **AutoRound+Rotate think 轻微增加**: think token 比 HF **增加 +6.7%**，think/answer 比 1.10，表现接近 HF
4. **Q4(描述春天) 所有模型思考未完成**: 达到 2048 max_new_tokens 上限仍未输出 `</think>`

### 关键发现

- GPTQ 量化后 think 显著变长（+45.5%），且 Q1 的 think 从中文变成英文结构化格式（Thinking Process），风格改变明显
- AutoRound 量化后 think 长度与 HF 接近（-13.1%），风格保持一致（中文思考为主），整体质量保持较好
- AutoRound+Rotate 量化后 think 长度与 HF 基本一致（+6.7%），表现稳定
- 量化对 answer 长度影响较小，GPTQ 的 answer 略短（因 think 占用更多 token 预算）

### Rotate 模型加载问题

非 selfgen 版本的 AutoRound+Rotate 模型存在双套 weight_map 问题（同时包含 CausalLM 和 VLM 两套键名），
需要在加载时修正 `model.safetensors.index.json` 仅保留 CausalLM 键，并修正 `config.json` 中的
`block_name_to_quantize` 从 `model.language_model.layers` 为 `model.layers`，否则模型输出乱码。
详见 `eval_chat_think.py` 中的 `_fix_dual_key_index()` 函数。
