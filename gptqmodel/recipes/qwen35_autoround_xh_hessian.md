# Qwen dense AutoRound: `int_sym_xh_hessian`

使用 `data_type="int_sym_xh_hessian"` 一项选择 Hessian 初始化和正 log-scale 优化。
它与 `int_sym_xh` 并列，默认参数由 GPTQModel/AutoRound 维护，ModelZoo 无需展开求解器配置。
底层保留 `int_sym_xh` 的 W4/G64、正 scale、零点 8 和现有 GPTQ 打包。
新模式改变初始化、scale 参数化和最佳参数选择，不是运行 GPTQ 的逐列误差补偿。
原 `optimization_mode="standard"` 路径不变。这是实验方法，尚无 27B 任务精度收益结论。

## 算法

1. 按 Transformer block 校准，给每个被量化的 Linear 搜集输入二阶矩。
   对每组 64 个输入通道保存 `H_gg = mean(x_g x_g^T)`，FP32 累加，按有效 token
   加权；排除 padding 和保留验证样本。同一 Linear 的所有输出行共享 H。
   它是线性重建目标 Hessian 的一半，不是整个 Transformer 的参数 Hessian。
2. 对每个输出行、每个 G64 分组搜索一个正 scale：
   `q = clip(round(w/s), -8, 7)`，最小化 `(s*q-w)^T H_gg (s*q-w)`。
   保留组内完整相关性，忽略跨组项；不是每个 weight 独立一个 scale。
   候选包括 XH 初值 `max(-min(w)/8, max(w)/7)`、正 GPTQ 初值
   `max(abs(w))/7.5`，以及 XH 初值乘 0.5 到 1.5 的 33 个等距系数。
   scale 先转为部署存储 dtype 再评分，始终保留 XH 候选。
3. 固定初始化 `s0`，训练每组 `rho` 和每权重 rounding 扰动 `v`：
   `s = cast_scale_dtype(max(s0*exp(rho), scale_threshold))`，
   `q = clip(round_STE(w/s + v), -8, 7)`。
   默认限制 `rho ∈ [ln(0.5), ln(2)]`，保证正 scale，允许增大和缩小。
   使用 AutoRound 的 SignSGD；rho 使用原 minmax_lr，v 使用原 rounding lr。
   训练目标仍是 **block 输出 MSE**，包含跨组、跨层及非线性影响。
4. 默认 256 个校准样本的最后 16 个仅用于验证，240 个用于 Hessian/梯度。
   每 25 步、末步及最终 optimizer step 后在相同验证样本上计算 token 加权 MSE，
   保存最优 rho/v。初始化本身参与 checkpoint 比较，验证 MSE 不优于它就保留初始化。
   padding 不计入验证分母。原 AutoRound 的训练 loss 归一化保持不变。
5. 使用被选中的 rho/v 还原 QDQ 权重，然后走原 AutoGPTQ pack/save。
   参数仍存储为 `qweight/qzeros/scales/g_idx`；GPTQ v1 packed 零点为 7，实际零点为 8。

若启用 AutoRound 的量化前缀输入，H 来自该 block 的实际量化前缀输入，teacher 输出
仍来自 FP 前缀；这继承原 AutoRound 的累计误差校正语义。搜索仅优化本层局部替代目标。
跨组 Hessian 项未保留，因此不能声称初始化的完整 Linear/Transformer MSE 必然下降。
统计矩阵按 G64 组织，无需求逆、阻尼或 GPTQ 的 Cholesky 因子。
例如 17408 输入通道的 Linear 统计量约 4.25 MiB，而完整 FP32 Gram 约 1.13 GiB。

## 使用

```python
from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

quantize_qwen35_autoround(
    model_dir="/path/to/Qwen3.8-27B",
    output_dir="/path/to/new-output",
    topology="dense", mode="llm-only",
    data_type="int_sym_xh_hessian", bits=4, group_size=64,
    dataset="/path/to/calibration.jsonl",
    nsamples=256, seqlen=2048, iters=400, batch_size=8,
    low_gpu_mem_usage=True, device="cuda:0", seed=42,
)
```

CLI 只需 `--data_type int_sym_xh_hessian`；直接使用 AutoRound 时也是同一 `data_type` 参数。
ModelZoo 在现有 AutoRound workflow 中只改一行：

```yaml
quant:
  data_type: int_sym_xh_hessian  # 原来为 int_sym_xh，其余字段不变
```

不在 workflow 中添加 `optimization_mode` 或 `xh_hessian_config`；误用会提示改成上述名字。
后端保留原 `data_type="int_sym_xh", optimization_mode="xh_hessian"` API 和
`xh_hessian_config`，用于兼容已有实验与后端调试。

由外层设置 `CUDA_VISIBLE_DEVICES` 限定单卡。选择名称、模式和实际默认配置写入 recipe provenance；
checkpoint 的 `data_type` 仍为 `int_sym_xh`，另记录 `requested_data_type=int_sym_xh_hessian`、
`optimization_mode=xh_hessian` 和实际配置，加载端无需认识新的整数格式。
`xh_hessian_report.json` 记录每层有效 token、搜索前后分数，以及各 block 的初始/最佳
验证 MSE、最佳迭代和训练/验证样本数。不要比较不同 block 的原始 Hessian 分数大小。

首版仅支持 dense、单卡、W4A16/G64、FP16 scale 存储、`iters>0`、无低位 head/MTP/
混合精度扩展、无 norm/bias tuning、无 alg_ext。错误组合直接拒绝。vision 保留浮点。
不新增校准语料；训练/验证分割是一项算法差异，和 A 对照时须明确记录。
scale 可缩小，因此优化后的网格不保证覆盖原始极值；覆盖端点是 XH 初始公式的性质。

## 验证边界

`tests/ci/test_xh_hessian.py` 检查含相关通道的 Hessian 评分、mask、候选保底、梯度、
最佳参数恢复和 recipe。`test_xh_hessian_quant.py` 使用保留 linear/full attention 的
小型真实 Qwen3.5 模型运行优化、打包和 GPTQModel 重载，检查整数载荷、scale、零点和 logits。
它不是预训练 27B 精度测试；完整模型 LCB、Marlin 数值以及 HMONNX 部署需另行验证。
