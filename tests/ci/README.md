# GPTQModel 项目 CI 与 Agent 规则

先看 [整体视图](overview.md)：流水线、实际覆盖、Artifactory block、AGENTS 与平台交接。

```text
tests/ci/
├── AGENTS.md / README.md / overview.md
├── run.sh / run.py / select_tests.py
├── pytest.ini / impact_policy.json
├── test_*.py / test_*.impact.json
├── prepare_block.py / prepare_ling_block.py
├── fixtures/                 # 发布清单、SHA、来源与裁剪说明
└── modelzoo/                 # 消费方用例的 impact 声明
```

本仓 CI 用例与声明相邻放置，`lane` 决定执行资源；无需在目录外增加集中清单或代理测试。
根目录及 `gptqmodel/recipes/` 的 AGENTS 保留各自作用域规则；生产代码仍位于 `gptqmodel/`。

本仓当前远端是 Houmo Gerrit。项目 CI 与本地共用 `tests/ci/run.sh`。
CI 同事负责平台任务、机器、环境、checkout 与调度；仓库提供测试入口和结果契约，不提供平台模板。
外层生成整个 patchset 的变更路径列表，入口只消费该列表；runner/selector 不调用 Git，
也不读取 `.git`，可运行于导出的源码目录。双方版本信息由外层提供并校验。
原 `.github/workflows/{unit_tests,release,code_quality}.yml` 依赖上游 ModelCloud
镜像、服务和 runner，其存在不代表公司流水线已上线。

## 参考 xh2modelzoo 的设计

参考仓库 `/data01/home/yujy/work/v2/xh2modelzoo` 的根 `AGENTS.md`、
`xhmodel_merak/AGENTS.md`、`docs/architecture/quantization-boundary.md` 和
`benchmark_test/ci_test_transformer5.16.0/{AGENTS.md,README.md,run.sh,select_tests.py}`。
对齐其 `CI_TEST_CHANGED_FILES_FILE` 外部列表入口；本仓不保留其本地 Git discovery 兜底。

| ModelZoo 的设计 | GPTQModel 的落点 |
| --- | --- |
| 一个当前入口，按变更选择测试 | 一个 `run.sh`，policy / cpu / quant / modelzoo 四个 lane |
| 根指引简短，子目录补约束，命令留 README | 根 AGENTS、recipes/AGENTS、tests/ci/AGENTS |
| 显式消费关系，不仅靠 Python import 推断 | 测试旁同名 `.impact.json`，共享输入选择所有声明的消费方 |
| 小模型保留实际结构，必要导出使用固定权重 | API/小 tensor/小模型 CPU 回归，加独立 GPU 量化重载 |
| producer 与 consumer 分开验证 | 本仓负责量化和产物，下游负责消费与 HMONNX 导出 |

采用 ModelZoo 的“全局 policy + 测试本地声明”分层，不依赖完整 import/family 推断器。
动态 recipe 和共享依赖由明确声明表达；代码没有生产端测试映射时直接报错。
原集中 `lanes.json` 已拆为同名声明，消除新增 recipe 连带其他 family 的粗粒度映射。
新增独立功能无需修改 selector、policy 或旧测试声明；已登记的本仓用例统一迁入 `tests/ci/`，
测试与声明相邻维护，上游未接入的测试继续保留原位置。

## 新功能怎样独立接入

一个新功能的典型变更：

```text
gptqmodel/recipes/new_family.py
tests/ci/test_new_family.py
tests/ci/test_new_family.impact.json
```

声明内容：

```json
{
  "version": 1,
  "lane": "cpu",
  "patterns": [
    "gptqmodel/recipes/new_family.py",
    "gptqmodel/models/definitions/new_family.py"
  ]
}
```

selector 只接受 `tests/ci/**/*.impact.json`，发现其他测试目录误放声明直接报错。
owner 固定为相邻同名 `.py`，声明不能通过
`tests` 字段指向其他测试。`lane` 可为 `cpu`、`quant`；可选 `nodes` 只追加本文件的
pytest node 后缀，见 [Qwen3-Next quant 声明](test_qwen3_next_autoround.impact.json)。
混有多类测试的迁入文件可登记明确的 node；新增不同资源类别的用例优先分文件。
`tests/ci/modelzoo/*.impact.json` 的 owner 是消费方当前 CI 目录中的同名测试，不写本地代理测试。

新增这三个文件只选新用例与 policy；不改变旧代码变更原先会选哪些测试。
若新模型还修改了公共注册入口 `models/auto.py`、loader/writer 或共享算法，就仍需回归
旧消费者。独立新增配置无需改旧配置，不等于修改共享实现可以不跑旧测试。

控制点：新 CI 用例必须带声明；孤立声明、未知字段、重复 JSON key、非法路径与大范围
`gptqmodel/recipes/*.py` 等 catch-all 均拒绝。`gptqmodel/`、量化示例和随仓 AutoRound 核心
变化必须命中 CPU/quant 的生产端测试，不能只挂一个下游测试算生产端覆盖，`--all` 也不绕过。
未变更的历史用例不被强制一次性接入；受保护的存量源码下次修改时需补声明/用例。
声明的语义覆盖仍需审查，选择器不能从名字判断测试是否真的验证了新算法。

声明修改会选择 owner，即使新 patterns 不再命中本次源码；本地 owner 保留时不能单删声明。
测试与声明一起 rename/delete 可正常处理，变更清单需保留两端。
共享 helper 必须由每个实际消费者声明；所有匹配取并集，不存在“新声明覆盖旧声明”的优先级。
只有 [impact_policy.json](impact_policy.json) 中列出的基础设施/公共核心触发全集。

## 流水线划分

以下编号只描述验证类别，平台任务命名、拆分和触发由 CI 同事决定。

| 建议任务 | 入口/触发 | 检查内容 | 状态 |
| --- | --- | --- | --- |
| CI-01 `gptqmodel-policy` | `--lane policy`；每个 patchset，含文档 | 声明、diff、rename、状态报告、下载校验 | 已实现 |
| CI-02 `gptqmodel-cpu` | `--lane cpu`；代码/配置命中 | 多 family recipe、校准输入、打包、W4/W5 head 保存重载 | 已实现 |
| CI-03 `gptqmodel-quant` | `--lane quant`；量化路径命中 | Qwen3.5 W4/W8 G64；Qwen3-Next XH 混合精度/MTP；Ling 原始/吸收 MLA、KDA/MoE、MTP 保留和重载 | 已实现；需 GPU 与各用例的固定输入 |
| CI-04 `gptqmodel-modelzoo` | `--lane modelzoo`；接口/格式改动，依赖升级强制 `--all` | 指定双方版本的 recipe/CLI、XH dtype、head、Ling MLA 消费契约 | 已实现；需消费方环境 |
| CI-05 `gptqmodel-nightly` | 定时/手动 | 多架构、rotation/MLA/MoE、质量与性能基线 | 后续接入 |
| CI-06 `gptqmodel-release` | 候选版本/手动 | wheel 构建、独立环境安装、recipe/第三方资源完整性、双方兼容集 | 后续接入 |

合入要求 CI-01、CI-02，以及变更命中的 CI-03/CI-04。未命中的 lane 执行 policy，
记录 `not-required; policy passed`；所需 lane 缺环境、失败或有 skip 不能报告通过。
长耗时全模型质量评估不进入每个文档 patchset。

Qwen3.5 smoke 只证明有限样本量化、packed 产物、重载与有限 logits；不覆盖全部模态，
也没有精度阈值。Qwen3-Next 最小模型保留两类 attention、MoE 和 MTP，检查保存前后
整数载荷与同 dtype logits 一致；不能代表所有架构。CI-04 不重复完整权重量化，也不执行
HMONNX block 导出/真机推理；部署格式改动仍需在对应 ModelZoo 版本额外跑 block export CI。
实际覆盖、明确缺口及新增量化能力的接入标准见 [量化覆盖矩阵](quantization-matrix.md)。

## 运行与环境

只需标准库与 pytest 的入口：

```bash
tests/ci/run.sh --lane policy
tests/ci/run.sh --lane cpu --dry-run
python tests/ci/select_tests.py --changed-files-file changes.txt
```

使用已准备好的环境：

```bash
PYTHON=/path/to/quant-env/bin/python tests/ci/run.sh --lane cpu --all
```

本次验证基线为 Python 3.12.13、Torch 2.8.0、Transformers 5.16.1、pytest 8.4.2，
本机解释器为 `/data01/home/yujy/miniconda3/envs/xhquant_55_tf_latest_20260828/bin/python`。
CI 使用固定环境/镜像，机器路径不作为默认值，也不每次 `pip install -U`。
`pyproject.toml` 的宽版本范围不等于所有新模型在每个版本都经过验证。
CPU 需要可导入本仓的运行依赖；quant 另需 CUDA 与后端依赖；modelzoo 使用已配置 SDK
的消费方环境。policy 只需 pytest。运行记录保存实际版本。

GPU 由调度器独占分配；runner 尊重 `CUDA_VISIBLE_DEVICES`，不固定物理 GPU：

```bash
export CUDA_VISIBLE_DEVICES=<allocated-device>
export GPTQMODEL_QUANT_GUARD_MODEL_DIR=/path/to/qwen35-dense-min2
PYTHON=/path/to/quant-env/bin/python tests/ci/run.sh --lane quant --all
```

fixture 必须有 tokenizer、config、浮点权重，且为保留相关 attention 类型的 Qwen3.5
dense checkpoint。测试自动裁成保留 linear/full attention 的 2 层，完整量化两层。
不传本地目录时使用 `model_zoo2/ci/gptqmodel-blocks-v1/` 中已固定 URL/SHA 的
Qwen3.5-0.8B 两层浮点 block；自定义下载则设置
`GPTQMODEL_QUANT_GUARD_ARCHIVE_URL` 和 `GPTQMODEL_QUANT_GUARD_SHA256`。
SHA-256 由产物发布者提供，不能临时信任同次下载
计算出的值。`GPTQMODEL_QUANT_GUARD_CACHE` 控制缓存目录；校验后用独立目录解包，
不复用上次未完成的内容。CPU/联调禁用 CUDA 与 HF 网络下载。

已准备 Qwen3.5 dense/MoE 与 Gemma4 dense/MoE 的最小代表 block，保留层数分别为
2/2/4/2，均通过真实 HF shape 与浮点前向检查。来源、裁剪命令、SHA、shared-KV
等特殊结构及尚未覆盖的量化场景见 [block 说明](fixtures/README.md)。

Ling 另外准备了原尺寸 KDA/dense＋KDA/MoE＋MLA/MoE 三层 block，并接入已有的 4 组
真实小模型量化回归。命中 Ling 时，CPU/quant 环境需 `fla-core`。未设置
`LING3_FLASH_MODEL_DIR` 时，自动下载固定 SHA 的 21 KB 代码包，仅含配置、模型源码和
原 README；本地生成确定性小模型，不下载预训练权重。`LING3_FLASH_CODE_CACHE` 可指定
缓存目录，默认 `work_dirs/ci-block-fixtures/source-cache/`，缓存校验通过后可离线复用。
也可显式设置 `LING3_FLASH_MODEL_DIR=/path/to/ling3-flash-min3`，目录需包含 config 和
两份模型源码；缺文件直接失败，不跳过或静默回退。固定清单见
[ling3-flash-code-v1.json](fixtures/ling3-flash-code-v1.json)。
使用 HTTP 代理的机器需将内网 Artifactory 的 `10.10.1.53` 加入 `no_proxy`/`NO_PROXY`。
原尺寸权重与缩小维度量化证据的区别见 [Ling 说明](fixtures/ling3-flash.md)。

跨仓库联调显式绑定 consumer checkout：

```bash
PYTHON=/path/to/modelzoo-env/bin/python tests/ci/run.sh \
  --lane modelzoo --changed-files-file changes.txt \
  --modelzoo-dir /data01/home/yujy/work/v2/xh2modelzoo \
  --gptqmodel-commit <producer-commit> --modelzoo-commit <consumer-commit>
```

`XH2MODELZOO_SOURCE_DIR` 可替代 `--modelzoo-dir`。runner 将 `GPTQMODEL_SOURCE_DIR`
指向当前生产端，优先使用本仓与 `third_party/auto-round`。
正式 CI 的外层脚本准备并校验干净且固定 commit 的双仓源码，runner 不查询工作区状态。
`--gptqmodel-commit` / `--modelzoo-commit` 仅记录外部版本信息，不解析 ref 或验证目录版本；
也可通过 `CI_GPTQMODEL_COMMIT` / `CI_MODELZOO_COMMIT` 提供。缺失时报告 `commit: null`
及 `commit_source: not-provided`，不能把外部标签当作本地校验结果。
消费方用例限于当前 `benchmark_test/ci_test_transformer5.16.0`，保留其 `-c` 与
`--confcutdir`，不继承历史 benchmark fixture。

## 选择和报告

- `--changed-files-file changes.json` 接受 JSON 字符串数组；`.txt` 使用每行一个路径。
  含换行的文件名使用 JSON；rename 必须含旧、新路径。空文本文件或 JSON `[]` 是已知空 diff。
- 文件列表由外层计算；传路径清单，不传 unified diff 或 name-status 输出。
  文件不可读、格式错误或声明无效直接失败，`--all` 也不绕过。
- 不提供列表表示未知变更，按完整登记集选择；本地和 CI 行为相同。
  不检查 Git 历史和工作区，也不把干净 checkout 当作空 diff。
- 支持 `CI_TEST_CHANGED_FILES_FILE` 和 `CI_TEST_FORCE_ALL=1`。
  外层必须传整个 patchset 的清单，rename 包含旧、新路径，不只取最后一个提交。
  已移除 `--base` / `--head` / `--modelzoo-ref`；不再读取 `CI_TEST_BASE_REF` / `CI_TEST_HEAD_REF`。
- 受保护代码无生产端测试声明直接失败；非受保护的未知输入保守选择全集。
  根指引/CI README 等未作为测试输入的文档只跑 policy，声明了的文档仍触发其 owner。
  声明文件本身不会触发全集；命中的所有消费者取并集。
- `--all` 只扩展当前 lane；`--dry-run` 仅记录 `planned`，不代表测试通过。

报告在 `artifacts/ci/<lane>/{run.json,junit.xml}`，`--report-dir` 可调整。JSON 包括
diff 来源、原因、pytest 命令、外部提供的双方 commit、依赖、耗时和退出码；GPU 另记设备与
fixture 输入。pytest 非零退出原样传播；所选用例缺失/跳过视为门禁未完成。
默认整组超时 CPU/联调 30 分钟、quant 90 分钟，可用 `--timeout` 调整。超时终止测试
进程组及量化子进程；完整日志由 Jenkins console 保留。

## 与 CI 同事的交接及规则分层

交接内容仅为：`run.sh --lane ...`、Python/依赖与 fixture 要求、整个 patchset 的变更输入、
双方版本、退出码及 `run.json`/JUnit 路径。机器标签、环境安装、双仓拉取、触发和 GPU
资源互斥由 CI 同事配置，diff 生成、版本和工作区校验也在外层完成。
本仓不放 Jenkinsfile，不修改平台任务或分支保护。

本次不添加 `.github/workflows/` 平台任务，也不保留另一套 Qwen 专用 selector。
项目 CI 的入口、用例、声明、fixture 与文档统一位于 `tests/ci/`。既有上游测试枚举器
排除整个 `tests/ci/`，避免把项目 policy/CPU 用例逐个分配给上游 GPU 任务。

根 [AGENTS.md](../../AGENTS.md) 放职责与验证要求；
[recipes/AGENTS.md](../../gptqmodel/recipes/AGENTS.md) 放参数、执行、producer 产物规则；
[本目录 AGENTS.md](AGENTS.md) 放 fixture、lane 接入与证据要求。模型设计与用法留在对应
文档，不复制全部 ModelZoo 规则或通用 agent 流程。

## 本轮验证记录

生产端基线 `cacf8ab3` 加本次工作树；消费端 `604debd185e165b50ce18c38fc7aa21cbec3d465`，
消费端原有未跟踪草稿目录已在运行报告中记录，未修改其源码。

| 验证 | 实际结果 |
| --- | --- |
| 外部 diff 输入与无 Git 入口 | 53 项 policy 通过；无 `.git` 源码目录真实执行，Git 调用拦截、空/缺失/错误输入、外部版本记录与消费方环境通过 |
| 目录迁移后的 CPU 完整登记集 | 222 项通过，约 145 秒，含当前 41 项 policy；无失败、无 skip |
| 目录约束与量化收集 | 本仓目标及声明均在 `tests/ci/`；11 组 quant 完整收集，本次目录调整未重新运行 GPU 量化 |
| Ling 缓存修复对照 | 使用修复前函数失败、当前函数通过，第二次加载正确创建吸收投影 |
| 发布时 policy 与 block 切片/校验 | 56 项通过，含当时 42 项 policy 与 14 项 block/scale 回归 |
| 初始 CPU 完整登记集 | 181 项通过，约 141 秒；包含当时 22 项 policy |
| ModelZoo 消费集 | 141 项通过，约 49 秒 |
| Qwen3-Next GPU | 3 组真实 AutoRound / 保存重载通过，连同 policy 约 63 秒 |
| 已知空 diff 的 modelzoo lane | 仅执行 policy，通过，无需消费方环境 |
| 4 种裁剪 block | 真实 HF key/shape 与 BF16 浮点前向通过；文本路径 |
| Qwen3.5 完整 GPU guard | W4/W8 × GPTQ/AutoRound 的 4 组全部通过，连同 42 项 policy 共 46 项、约 228 秒；实际量化、载荷与重载前向 |
| Ling CPU | recipe、3 层裁剪、同名 checkpoint 的 HF 类缓存回归，连同 policy 共 60 项通过，约 10 秒 |
| Ling MLA 原有 CPU 回归 | 浮点 prefill/cached decode、非法布局、HF 浮点重载等 13 项通过；GPU 两项单独验证 |
| Ling GPU | 原布局 AutoRound（MTP 0/1）及吸收布局 GPTQ/AutoRound，4 组通过，连同 policy 共 46 项、约 135 秒 |
| Ling 改动的 ModelZoo 消费集 | XH dtype 与 Ling MLA 消费，106 项通过，约 33 秒 |
| 静态检查 | 新增 Python 的 Ruff、语法、shell 语法、Git diff 检查通过 |

这些结果不包含全模型精度、HMONNX 导出、真机推理或 release wheel 验证。

Ling 代码包默认下载接入后，在未设置 `LING3_FLASH_MODEL_DIR`、空归档/HF 模块缓存、
禁用 CUDA 和 HF 网络下载的独立评审 checkout 上，全量 CPU 241 项通过、0 skip，
约 142 秒。另验证了已有归档缓存时离线加载通过；本次没有重跑 GPU 量化。
报告位于 `artifacts/ci/ling-code-fixture/full-direct/cpu/`。

目录迁移后的报告和 Ling 修复对照日志位于 `artifacts/ci/qtl491-layout/`。
外部 diff 输入调整的报告位于 `artifacts/ci/qtl491-external-diff/`。

本轮 block/guard 报告为 `/tmp/gptqmodel-ci-block-validation-final/{cpu,quant}/run.json`
与同目录 JUnit；浮点 shape/forward 日志为 `/tmp/gptqmodel-block-forward.log`。
初次新增 scale 断言误拒绝源权重约 `1e-37` 通道的 FP16 下溢，现已按原始 group 值校验，
允许存储精度下本就为零的 group，并有反例保证正常权重不能被零 scale 擦除；最终四组重跑通过。

Ling 补充验证报告在 `/tmp/gptqmodel-ci-ling-final/{cpu,quant,modelzoo}/`。原尺寸 Ling
3 层 block 的结构与 BF16 前向通过；这些量化回归的实际输入是缩小维度的真实代码模型。
同名 checkpoint 的 HF 类缓存问题由组合量化回归发现并修复，详见 [Ling 说明](fixtures/ling3-flash.md)。

语法参考：[pytest 文件/node ID](https://docs.pytest.org/en/stable/how-to/usage.html)。
