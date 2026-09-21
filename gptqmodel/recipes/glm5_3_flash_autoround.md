# GLM-5.3-Flash AutoRound W4 G64 / W8

公开入口：`gptqmodel.recipes.glm5_3_flash_autoround.quantize_glm5_3_flash_autoround`。
AutoRound 优化每个原生 decoder block；非共享专家 W4 G64，其余选中的投影 W8 G64。
保留原生 MoE 路由及 gate/up clamp，KDA decay/beta 和 DSA 未吸收 K/V 因子保持浮点。
视觉、head、router、indexer、mHC、卷积和 DSA 因子交由部署端量化为 W8。

```bash
conda activate xhquant_55_tf_latest_20260828
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python examples/quantization/examples/example_glm5_3_flash_autoround.py \
  --model-dir /path/to/GLM-5.3-Flash \
  --prepared-model-dir /path/to/glm53-bf16-input \
  --output-dir /path/to/glm53-autoround-w4g64 \
  --calibration-jsonl /path/to/NeelNanda-pile-10k.jsonl \
  --device-map 0 --iters 200 --nsamples 128 --seqlen 2048 \
  --batch-size 1 --gradient-accumulate-steps 2
```

默认单卡、`nblocks=1`、128 条 2048-token 样本、200 次优化、有效 batch size 2。
与累积 8 的配置相比，累积 2 将每个 block 的训练前后向次数从 1600 降到 400；
校准、权重加载、打包和导出的开销仍存在，总耗时须实测。每步有效 batch 随之变化，
最终精度通过量化产物验证。仅设置 `nsamples=128, iters=200` 不会减少原配置的计算量。
API 的
`gradient_accumulate_steps` 可在减小 batch size 时保持有效 batch size。
`device_map="0,1,2,3"` 可选择四张可见卡，是否需要多卡应按实际配置验证。
FP8 准备按 shard 流式转为 BF16，本次完整模型中间权重约 628 GB（585 GiB）；初始 HF 加载仍需整模主机 RAM。
AutoRound 启用 CPU/GPU 低内存选项及逐 block 即时打包保存。导出、HMONNX 和 runtime 属于 ModelZoo。
单卡方案将专家原权重留在 CPU，由 AutoRound 的投影读取逻辑按需搬入 GPU；
专家量化中间量使用非重入梯度检查点重算，减少前向留存。该模式增加传输和重算开销，
显存结论须以实际模型、batch size 和序列长度的验证结果为准。

校准按 seed 打乱文本，选够 `nsamples` 条完整 `seqlen` 的非重复样本，不补 padding 凑数；
不足时报错。输出记录实际 token SHA、样本数、损失检查次数、优化参数、后端版本及模块清单审计。
非有限损失立即失败；校验通过才原子发布目标目录。失败时保留隐藏 staging 目录及已写入的权重，
异常附带路径；该目录尚未通过验收，不能直接视为完整产物。AutoRound 原生保存会省略
默认 `iters=200`，recipe 按后端默认语义恢复该字段，显式非正迭代数仍拒绝。
原模型的 `top_p=0.95` 与缺失的采样开关由 AutoRound 原生 helper 对齐后保存，
避免 Transformers 严格校验拒绝序列化；部署 demo 显式使用贪心生成。
量化返回前释放 AutoRound 回调引用和 CUDA 缓存，便于同进程继续导出。

2026-09-12 的真实前 5 层单卡探针覆盖 KDA+dense、CSA/DSA+MoE、KDA+MoE：
16 条 2048-token 样本、2 次迭代、batch 1、累积 8，完成优化、保存、审计和独立进程重载。
CUDA 分配峰值 72.0 GiB、保留峰值 78.4 GiB，总计约 503 秒；
审计为 1728 个 W4 和 43 个 W8 投影。该证据验证资源及保存流程，整模质量验收另行记录。

2026-09-15 的同尺寸 5 层探针改为 128 条 2048-token 样本、batch 1、累积 2、2 次迭代，
在单张 A100 80GB 上完成优化、保存、清单审计和独立进程重载，logits 全部有限。
CUDA 分配峰值约 72.0 GiB、保留峰值约 78.4 GiB；CSA/MoE 和 KDA/MoE 的每次迭代
前后向及优化器操作合计分别约 18.7 秒和 17.2 秒，不含校准、最佳参数复制和打包。
这些是短测数据，200 次迭代的全模型耗时与质量须另行验收。

产物使用 `auto_round:gptqmodel` 格式。`quant_method=gptq` 是存储格式标识，
`provider=auto-round`、`autoround_version`、`iters` 和 `glm53_quantization_recipe.json`
保留实际算法。量化没有调用 GPTQ 求解器；整数载荷不因 metadata 规范化而改变。
输出包含视觉资源及完整语言主干，MTP 不参与 45 层部署。

验证用例：`tests/ci/test_glm53_autoround.py` 检查 clamped experts、精度规则、校准和失败契约；
`tests/ci/test_glm53_autoround_quant.py` 用两层原生合成模型执行真实优化、即时保存、
独立进程重载、packed 权重一致性和有限 logits。完整模型质量需另行通过实际部署验证。
