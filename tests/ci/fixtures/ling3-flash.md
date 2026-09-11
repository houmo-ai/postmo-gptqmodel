# Ling-3.0-flash block 与量化回归

Ling 要单独覆盖 KDA、MLA 吸收和 MoE。Qwen 的 linear attention 不能替代 Ling 的
KDA/ShortConvolution/cache 合约，原 MLA 权重与 `xh2_absorbed_v1` 也属于不同产物布局。

## 可以复用的 ModelZoo 小模型

ModelZoo 当前 CI 的 `fixtures/ling3_flash/` 已有固定源码的一层 MLA＋MoE 模型：
hidden=64、32 heads、2 routed experts 和 1 shared expert，不构造 KDA、没有预训练权重。
它验证普通/分层加载、融合投影的整数载荷保留、独立 K/V cache 等消费行为，继续由
`modelzoo` lane 调用其原用例；不把消费方代码反向 import 到生产端。

固定源码基线来自 `inclusionAI/Ling-3.0-flash` 的
`e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3`。本地原模型的两份源码 SHA 与该基线相同：

| 文件 | SHA-256 |
| --- | --- |
| `configuration_bailing_moe_v3.py` | `f2c048966aec8a2f042cfeb1351f74d51a28589b409c55baae7d24e841c1f6c4` |
| `modeling_bailing_moe_v3.py` | `c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40` |

ModelZoo 自带建模源码为延迟 FLA 导入做过明确修改；其 `upstream.json` 另记修改后的 SHA。

## 已裁出的原尺寸浮点 block

来源 `/data01/datasets/Ling-3.0-flash`，输出
`work_dirs/ci-block-fixtures/ling3-flash-min3/`，原层号从 0 开始：

| 原层 → 新层 | 保留机制 |
| --- | --- |
| 0 → 0 | KDA＋dense MLP |
| 2 → 1 | KDA＋routed/shared MoE |
| 5 → 2 | MLA＋routed/shared MoE |

同步设置 `num_hidden_layers=3`、`layer_group_size=3`、`first_k_dense_replace=1`，保证
重建后仍是 KDA/KDA/MLA 和 dense/MoE/MoE。仅取前两层会漏掉 MLA 与 MoE；直接重排层号
而不改这些配置会加载错误的 attention/MLP 结构。

保留 hidden=2560、32 heads、512 experts、top-8、8 个路由组、top-4 group、shared expert、
router bias、原词表与权重。3126 个张量的真实 HF key/shape 与 BF16 文本前向通过。
权重为 13.159 GiB，SHA-256：
`69cdc0a56cbd64d64f25ba79b4afb1087e3307595fc6fb26db41b8d64833fa3f`。
完整来源/源码摘要和层映射保存在 block 内的 `ci_block_provenance.json`。

这是保留三种基础层机制的最小原尺寸代表，不覆盖所有层位置、后段 SwiGLU 限幅组合、
长上下文或完整路由分布。短文本前向也不等于 512 个 expert 全部被激活或量化。
默认不保留 MTP；`--keep-mtp` 可额外复制原第 42 层为新第 3 层，且不改变其结构。
本轮原尺寸验证为 3 层版本；MTP 参数的切片另有小 tensor 回归。

```bash
python tests/ci/prepare_ling_block.py --source /path/to/Ling-3.0-flash --plan
/path/to/quant-env/bin/python tests/ci/prepare_ling_block.py \
  --source /path/to/Ling-3.0-flash --output /path/to/ling3-flash-min3
# 从仓库根执行，使用已安装本仓及 fla-core 的环境。
PYTHONPATH="$PWD:$PWD/third_party/auto-round" /path/to/quant-env/bin/python \
  tests/ci/prepare_ling_block.py --source /path/to/ling3-flash-min3 --check-shapes
CUDA_VISIBLE_DEVICES=<allocated-device> PYTHONPATH="$PWD:$PWD/third_party/auto-round" \
  /path/to/quant-env/bin/python tests/ci/prepare_ling_block.py \
  --source /path/to/ling3-flash-min3 --check-forward
```

## 已接入的生产端门禁

复用本仓现有真实 remote-code 小模型测试，自动登记各自的 `.impact.json`：

| 用例 | 组合与断言 |
| --- | --- |
| `test_ling3_flash_autoround.py::test_real_ling_autoround_quantize_save_load` | 原 MLA 布局 AutoRound，两种 MTP 配置；`int_sym_xh`、attention W8 / routed expert W4、G64，检查浮点排除与保存重载 |
| `test_ling3_flash_mla.py::test_real_absorbed_quantize_load_resave` | `xh2_absorbed_v1`，GPTQ/AutoRound 各一组；吸收后的投影打包、MTP 浮点载荷保留、重载/再保存整数一致、MLA prefill/decode 一致 |

这些量化测试用固定 seed 初始化缩小维度的真实模型，包含 KDA/MLA/MoE，实际执行后端，
与上面的原尺寸预训练权重 block 分开记录。它们不能被描述为 13 GiB 原尺寸 block 已完成量化。
KDA 的真实 CUDA/FLA 路径在校准时执行；后续若改动 KDA cached decode，仍需补该路径专用断言。

CPU 加载回归与 quant 测试共用同一个源码入口。未设置 `LING3_FLASH_MODEL_DIR` 时，
自动下载 [ling3-flash-code-v1](http://10.10.1.53:8081/artifactory/model_zoo2/ci/ling3-flash-code-v1/ling3-flash-code-v1.tar.gz)，
压缩后 21,570 字节，只包含发布 block 中原样保留的 config、两份 Python 源码与 README，
不含权重和 tokenizer。归档及文件 SHA 固定在 [清单](ling3-flash-code-v1.json)，下载与
解包后均校验；归档按 SHA 缓存，每次使用单独解包目录。本地确定性生成小权重和 tokenizer。

`LING3_FLASH_CODE_CACHE` 指定缓存目录，默认 `work_dirs/ci-block-fixtures/source-cache/`。
离线机器可预先放入清单 SHA 命名的 `<sha256>.tar.gz`，或显式设置
`LING3_FLASH_MODEL_DIR` 为包含上述三份配置/源码的本地目录。显式目录缺文件、下载失败、
摘要不符均直接失败，不自动跳过。日志记录实际目录或下载 URL/SHA。
HTTP 代理环境需将 `10.10.1.53` 加入 `no_proxy`/`NO_PROXY`，让内网请求直连。
本轮环境 `fla-core=0.5.2`、Torch 2.8、Transformers 5.16.1；CPU 不运行 CUDA KDA，
quant 必须有 CUDA，任何选中用例 skip 都不能形成绿色门禁。

```bash
CUDA_VISIBLE_DEVICES=<allocated-device> \
  PYTHON=/path/to/quant-env/bin/python tests/ci/run.sh --lane quant --all
```

本轮还修复了一个被组合运行暴露的问题：同名 `out` checkpoint 的配置源码相同时，HF
可能复用先前未吸收的模型类。Ling 兼容层现在为已加载类安装按配置启用的 MLA 构造/重载逻辑，
并有“原布局 → 吸收布局 → 原布局”的真实 meta 构造回归；不清缓存或隔离进程来隐藏该问题。

具体地，`install_ling_mla_class` 使缓存类在 `ling_mla_layout=xh2_absorbed_v1` 时
创建 `q_absorbed_proj` 等吸收投影；原布局继续走原构造/forward，MTP 层排除吸收。
`install_ling_mla_model_class` 则在吸收模型重载后恢复派生 RoPE 状态并校验布局。
两种安装都有重复调用保护。这是 checkpoint 加载兼容修复，不是测试专用分支。

已再次用同一回归测试做对照：仅把 `patch_ling_3_flash_remote_code` 换回修复前版本，
第二次加载的 attention 缺少 `q_absorbed_proj`，断言失败；当前版本通过。
日志位于 `artifacts/ci/qtl491-layout/ling-{without,with}-fix.log`。

本轮报告目录 `/tmp/gptqmodel-ci-ling-final/`；原尺寸 block 的 CLI 验证日志为
`/tmp/gptqmodel-ling-block-forward-final.log`。
最终 CPU 60 项通过，GPU 4 组量化连同 policy 共 46 项通过（约 135 秒），
ModelZoo 消费集 106 项通过；全部保留真实失败/跳过状态检查。
原有 MLA 浮点/加载 CPU 回归另外 13 项通过。输入来源、源码 SHA 与原尺寸 block 指纹
汇总在报告目录的 `inputs.json`。
