# GPTQModel

## 仓库职责

- 本仓库是 xh2modelzoo 使用的权重量化生产端。负责公开 recipe、HF 加载、校准、
  GPTQ/AutoRound 求解、量化侧结构变换、打包和 checkpoint 保存/重载。
  ModelZoo 负责部署配置映射、产物消费、图与 runtime；xhquant 负责图量化和目标导出。
  不反向依赖 ModelZoo 的 workflow、SDK 或示例目录。
- 修改跨仓库接口先读 [CI 与协作边界](tests/ci/README.md)，涉及 recipe 同时遵循
  [recipe 约定](gptqmodel/recipes/AGENTS.md)。校验真实下游调用方，不能只检查函数签名。
- 量化算法、checkpoint 格式、存储位宽、逻辑位宽分别描述。AutoRound 写出的 GPTQ
  格式不代表运行了 GPTQ；W5 的 8 bit 存储不能被当成 W8 数值精度。
- 量化前的 rotation/MLA 吸收属于生产端；改动后核对模块树、校准顺序、配置与权重布局。
  head 补量化须保留主体载荷和 producer 信息，tied head 解绑定须能保存并正确重载。
- `third_party/auto-round/` 是随仓维护的后端；修改时核对调用它的 recipe、CLI 和产物。
  不用另外安装的上游 AutoRound 掩盖本地后端问题。

## 验证入口

- 当前项目 CI 入口是 `tests/ci/run.sh`，选择预览用 `--lane cpu --dry-run`。
  流水线类别、运行环境和完整命令见 [README](tests/ci/README.md)。
  `.github/workflows/` 中的上游环境配置不代表 Houmo Gerrit/Jenkins 已经接通。
- CI 入口只消费外层传入的变更文件列表与版本信息，不调用 Git，不读取 `.git`，
  不扫描 staged/unstaged/untracked 来推断 diff。checkout、patchset diff 与版本校验由外层负责。
- 项目 CI 的入口、测试、同名 `.impact.json` 和资源清单统一放在 `tests/ci/`。
  纳入项目 CI 的已有本仓用例与声明一起迁入，不在 `tests/` 其他目录散放 CI 声明。
  新增独立模型/功能不改其他用例声明或全局 policy。
  声明实际实现、配置和共享输入，不能用 `recipes/*.py` 等大范围模式覆盖新模型。
  不批量搬迁上游 `tests/models/`，不扫描全仓测试分配 GPU。
  测试设计遵循 [测试约定](tests/ci/AGENTS.md)。
- 新增量化算法、精度、结构变换或产物布局，按 [量化覆盖矩阵](tests/ci/quantization-matrix.md)
  补真实量化/保存重载用例；复用结构的模型接入可复用算法代表集，但必须有本模型的功能测试。
  CI 平台任务、机器、环境和调度由 CI 同事管理；仓库只交付执行入口、测试及报告契约。
- 修改公共 recipe、配置或权重格式时，除生产端测试外，运行指定 ModelZoo checkout 的
  `modelzoo` lane；记录外层提供的双方 commit 和实际依赖版本，未提供的 commit 记为未知。
  单仓 impact 不能发现另一仓的变更。
- 报告实际通过/失败/跳过的用例与未执行项。dry-run、文件存在和接口签名通过不能宣称
  量化正确、可重载、导出正确或真机支持；环境缺失不能记为通过。

## 定位与维护

存在 `.codegraph/` 时先用 CodeGraph 定位代码与影响范围；未覆盖内容再用 `rg` 或直接读取。
本机 CLI 不支持 `explore` 时使用可用的 MCP 或 `context/query`，不自动重建索引。
保留已有暂存和未提交改动，不恢复与当前任务无关的文件。

## 提交与推送

- 采用 `type(scope): [Refs: JIRA-ID] subject`，英文祈使句，无句末句号。
  Jira 使用用户指定编号；仅确实关闭问题时使用 `Closes`，不猜测关联号。
- 参考 [公司提交规范](https://houmo.feishu.cn/wiki/XZjGwHlMfinyw5kPmKAcxkrpnDH)。
  公司规范标题上限为 72 字符；本仓标题控制在 50 字符以内，避免 Gerrit 长标题警告。
  正文每行不超过 72 字符，段落手动换行，标题与正文、正文与 footer 之间留空行。
- 正文说明具体改动、实际验证及未覆盖范围；无关改动分开提交。
  模型权重、上传凭据和运行缓存不进入 Git。
- 提交前检查 message 草稿；commit/amend 后再次逐行检查 `git log -1 --format=%B`
  的实际内容，包括 hook 修改后的长度、Jira 与 footer。`git diff --check` 不检查 message。
  发现超长行先修正再 push，不忽略本地或 CI 的格式警告。
- 新评审由 commit-msg hook 生成 Change-Id；更新现有评审必须保留原 Change-Id。
  本仓推送到 `origin HEAD:refs/for/develop`（其他目标按用户指定），不使用 force push。
  只更新当前待审提交，不为格式问题重写已合入历史。

`AGENTS.md` 只保留仓库特有约束与阅读入口。设计和命令放在对应 README；不复制 ModelZoo
的冻结目录、芯片规则或特定模型接入政策到本仓。更新接口时一起维护相关链接与验证说明。
