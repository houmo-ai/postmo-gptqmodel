# 最小 block：来源、裁剪与验证

先复用 ModelZoo 的浮点 block；结构不够时从本地原模型裁剪。已量化 block 用于消费/重载，
不能再次当作真实量化测试的浮点输入。权重放缓存或资源库，不提交 Git。

## 已经准备好的代表集

以下均为真实 checkpoint 权重，层号从 0 开始。最小化的是保留指定机制的**文本层数**，
hidden size、head、expert 数与 top-k 不缩减；原 tokenizer 和视觉/音频辅助权重保留，
以维持公开 recipe 的 HF 加载结构。本表的验证范围为文本路径，不能据此声称覆盖多模态量化。

本地输出目录为 `work_dirs/ci-block-fixtures/`：

| 输出子目录 | 浮点来源 | 保留源层 | 保留机制 | 权重大小 |
| --- | --- | --- | --- | --- |
| `qwen35-dense-min2` | ModelZoo Qwen3.5-0.8B 浮点 4 层归档 | 0、3 | linear + full attention | 0.735 GiB |
| `qwen35-moe-min2` | `/data01/datasets/Qwen3.5-35B-A3B` | 0、3 | 两种 attention、256 routed experts、top-8、shared expert | 5.854 GiB |
| `gemma4-dense-shared-min4` | `/data01/datasets/gemma-4-E2B-it` | 13、14、15、19 | sliding/full 的 KV donor 与 borrower 各一层、per-layer input | 2.536 GiB |
| `gemma4-moe-min2` | `/data01/datasets/gemma-4-26B-A4B-it` | 0、5 | sliding + full attention、128 experts、top-8 | 5.502 GiB |
| `ling3-flash-min3` | `/data01/datasets/Ling-3.0-flash` | 0、2、5 | KDA/dense、KDA/MoE、MLA/MoE，保留 512 experts | 13.159 GiB |

5 个 block 均已按 Transformers 5.16.1 的真实模型检查全部 key/shape，并以 BF16 执行
文本前向，输出形状与有限性通过。MoE 保留全部 expert 不代表短文本激活了每个 expert；
后续 MoE 量化测试还须检查遍历、校准与路由覆盖。

Ling 的 ModelZoo 小模型复用、独立裁剪命令、MLA 吸收和 MTP 门禁见 [Ling 专项说明](ling3-flash.md)。

Gemma E2B 不能直接取“前两层”：原模型前 15 层提供 KV，后 20 层共享 KV。
13/14 是两种 attention 的最后 donor，15/19 是相应 borrower。
裁剪后重排成 4 层、`num_kv_shared_layers=2`，同时切分 per-layer embedding 的列与
projection 的行；源 checkpoint 中 borrower 的冗余 K/V 权重去掉。
MTP 不在这些文本 block 的覆盖范围；现有 Qwen3-Next 真实量化测试单独覆盖 MTP。

## Artifactory 发布资源

已使用 `/data01/modelzoo2/upload.sh` 发布至实际仓库 `model_zoo2` 的
`ci/gptqmodel-blocks-v1/`，共 5 个浮点归档，压缩后合计 21.68 GiB。
每个归档保留 checkpoint、tokenizer、所需配置/源码和裁剪 provenance。
所有上传文件的服务端 SHA-256 与大小均已核对；Qwen dense 另经默认 CI 下载入口
重新下载、校验、解包和 HF shape 检查，权重摘要与量化已通过的本地 block 一致。

| 资源 | 下载 | 压缩大小 |
| --- | --- | ---: |
| Qwen3.5 dense，2 层 | [qwen35-dense-min2-fp-v1.tar.gz](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/qwen35-dense-min2-fp-v1.tar.gz) | 0.60 GiB |
| Qwen3.5 MoE，2 层 | [qwen35-moe-min2-fp-v1.tar.gz](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/qwen35-moe-min2-fp-v1.tar.gz) | 4.40 GiB |
| Gemma4 dense shared-KV，4 层 | [gemma4-dense-shared-min4-fp-v1.tar.gz](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/gemma4-dense-shared-min4-fp-v1.tar.gz) | 1.70 GiB |
| Gemma4 MoE，2 层 | [gemma4-moe-min2-fp-v1.tar.gz](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/gemma4-moe-min2-fp-v1.tar.gz) | 4.43 GiB |
| Ling3.0 Flash，3 层 | [ling3-flash-min3-fp-v1.tar.gz](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/ling3-flash-min3-fp-v1.tar.gz) | 10.56 GiB |

服务端同时提供 [manifest.json](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/manifest.json)、
[SHA256SUMS](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/SHA256SUMS) 和
[README](http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1/README.md)。
仓库内固定同一份 [发布清单](artifactory-v1.json) 与 [归档摘要](artifactory-v1.sha256)，
CI 使用受版本控制的摘要，不以远端最新摘要替代。清单记录归档与内部文件摘要、原层号、
来源和验证范围；发布内容变化时使用新版本目录，不能覆盖 v1。

单包下载示例，先设置 GPTQModel checkout 和缓存目录：

```bash
set -euo pipefail
GPTQMODEL_REPO=/path/to/gptqmodel
CI_BLOCK_CACHE=/path/to/ci-block-cache
CI_BLOCK_URL=http://10.10.1.53:8081/artifactory/model_zoo2/ci/gptqmodel-blocks-v1
CI_BLOCK_ARCHIVE=qwen35-dense-min2-fp-v1.tar.gz
mkdir -p "$CI_BLOCK_CACHE"
cd "$CI_BLOCK_CACHE"
curl --fail --location --noproxy 10.10.1.53 -O "$CI_BLOCK_URL/$CI_BLOCK_ARCHIVE"
sha256sum --check --ignore-missing "$GPTQMODEL_REPO/tests/ci/fixtures/artifactory-v1.sha256"
tar -xzf "$CI_BLOCK_ARCHIVE"
```

按表替换归档名即可下载其他 block；解包目录与 block id 相同。
Qwen dense guard 已固定本批 URL/SHA，可自动缓存下载。Ling 的 CPU/quant 小模型用例
默认只下载独立的 [21 KB 代码包](http://10.10.1.53:8081/artifactory/model_zoo2/ci/ling3-flash-code-v1/ling3-flash-code-v1.tar.gz)，
固定摘要见 [代码包清单](ling3-flash-code-v1.json)；不需要下载上表中的 Ling 权重包。
`LING3_FLASH_CODE_CACHE` 可指定归档缓存，校验通过后支持离线复用；每次使用独立解包目录。
显式设置 `LING3_FLASH_MODEL_DIR` 则使用本地 config 和两份模型源码，缺文件直接失败。
已核对原始目录与该 block 生成的 3 种 MTP/head 配置及全部初始化张量一致，
这不代表现有小模型量化测试会量化 block 的原尺寸权重。

本轮发布与回读证据位于 `artifacts/ci/block-publish/`，其中 `result.json` 记录发布结果，
`qwen-default-readback.json` 记录默认下载路径验证，`ling-code-reuse.json` 记录源码/配置复用验证。

## 现有资源怎样复用

ModelZoo 的 `benchmark_test/ci_test_transformer5.16.0/_transformer516_block_export.py`
已定义 4 个 W4/G64 AutoRound block：Qwen3.5-9B 4 层、Qwen3.5-35B-A3B 2 层、
Gemma4 E2B 6 层、Gemma4 26B-A4B 2 层，包含固定 SHA 与消费端校验。
继续由 ModelZoo 的导出/消费 CI 使用，不把其 helper 反向 import 到 GPTQModel。

同一资源库的 `transformer55_ci_trimmed_simple_v2` 中有 12 个浮点裁剪归档。
下列 SHA 来自已有 Artifactory 存储元数据；本轮实际下载并校验了第一项，其余只核对元数据：

| 归档名（共同前缀 `transformer55-ci-`） | SHA-256 |
| --- | --- |
| `Qwen3.5-0.8B-layers4.tar.gz` | `88352a0ea5ae449a0cd4876cb4b159b27ac42317b0fa977a9f95ebcf142491ed` |
| `Qwen3.5-9B-layers4.tar.gz` | `7f5ae2ded93b616e3a06a0ee876423948d2964088481c17b1295c5f8b9028255` |
| `Qwen3.5-35B-A3B-layers2.tar.gz` | `7cf78c06dcca8cc4a16a7bb89fc00d0536bd338c73521fb62442f218b2e45c67` |
| `gemma-4-E2B-it-layers6.tar.gz` | `13e727391d606fe95343ff11a1bd2965b074e596bd42f4735e5bb8a37728813f` |
| `gemma-4-26B-A4B-it-layers2.tar.gz` | `308c87d1e3779b4c760fddcbe38913edc52e4fe4b1e026c3a7f7f0eddbe54572` |

下载根路径为 `http://10.10.1.53:8082/artifactory/model_zoo2/transformer55_ci_trimmed_simple_v2/`。
这些归档保留为裁剪来源。当前 Qwen dense guard 默认使用本批发布到 `ci/` 的 2 层 block；
内网访问须使用能直连资源库的网络/代理配置。自定义 URL 必须提供对应 SHA，或显式传入本地浮点模型。

## 自己裁剪

使用固定量化环境的 Python。`--plan` 只读配置，不依赖 Torch：

```bash
python tests/ci/prepare_block.py --source /path/to/original-model --plan
/path/to/quant-env/bin/python tests/ci/prepare_block.py \
  --source /path/to/original-model --output /path/to/new-block
/path/to/quant-env/bin/python tests/ci/prepare_block.py \
  --source /path/to/new-block --check-shapes
```

支持明确的 Qwen3.5 dense/MoE 与 Gemma4 配置。拒绝量化权重、缺失分片、重复 tensor、
缺失 attention 类型、缺失 KV donor、尚未实现的 sparse-step 重排，以及覆盖原目录。
输出记录 `ci_block_provenance.json`：来源路径、源 config SHA、分片大小、保留层、输出权重
SHA/大小与排除的机制。源分片大小不是内容校验值；跨机器发布/复用需固定整个归档 SHA。
`--check-shapes` 用 meta device 核对真实 HF 模块树，只豁免实际 tied weight 的重复存储。

Qwen guard 对输入再次取代表层并量化全部两层，没有 `max_quant_layers=2` 截断四层模型的漏洞。
其他 block 已具备形状/浮点前向证据；在相应真实量化用例接入前，不能写成 GPTQ/AutoRound
量化已通过。新增机制继续新增自己的测试和 `.impact.json`，不用改旧测试的声明。

## 本轮输出指纹

| block | `model.safetensors` SHA-256 |
| --- | --- |
| Qwen dense | `a2ad60a724c23f13d79961a7ca702582e2cc573487eeb6810bd328afd43b2f6f` |
| Qwen MoE | `fb8a8b38eb5139de4051490e668d6ac0a3693e36808d393dc07702ab90133ee9` |
| Gemma dense shared-KV | `77d221ecdafb8b870e7ea6efdd12543185c7b1cf223b661e4f4f4bd781c8db96` |
| Gemma MoE | `5ccf3795b790f2ad908f9c52dacca43aceb2af2dfac861abfa1e15d94a128c5f` |
| Ling3.0 Flash | `69cdc0a56cbd64d64f25ba79b4afb1087e3307595fc6fb26db41b8d64833fa3f` |

本地完整 shape/forward 日志为 `/tmp/gptqmodel-block-forward.log`。
量化产物证据与缺口见 [量化覆盖矩阵](../quantization-matrix.md)。
