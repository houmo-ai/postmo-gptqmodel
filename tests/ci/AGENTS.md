# GPTQModel 项目 CI

先读 [README.md](README.md)。新增 CI 用例放在本目录，提交同名 `.impact.json` 自动发现；
声明只拥有相邻同名测试，可用 `nodes` 选定其中的函数。已接入的本仓用例也与声明一起迁入，
入口、测试、声明及资源清单全部在 `tests/ci/` 内维护，不在其他测试目录留副本或代理。
ModelZoo 消费用例的声明单独放 `modelzoo/`，文件名指向其当前 CI 中的同名测试。

- 新增独立功能只新增实现、测试和声明，不修改旧声明、selector 或全局
  [impact_policy.json](impact_policy.json)。共享输入在每个实际消费方声明中列出；
  注册入口/公共量化核心改动继续回归已有消费者，不能用“增量”绕过。
- 新 CI 测试无声明、声明无测试、受保护代码没有生产端测试映射均失败；`--all` 不绕过。
  本仓 CI 测试或声明放到 `tests/ci/` 之外也失败；ModelZoo 消费方测试仍归消费仓库所有。
  修改声明必跑其 owner；测试与声明一起移动/删除，不单独删声明隐藏回归。
  全局 policy 只列基础设施和真实公共核心的路径，不能用模型目录通配符作为新功能接入入口。

- `policy` 只依赖 Python 标准库与 pytest；`cpu` 使用已安装的量化环境、小 tensor 和
  本地生成的最小模型，不下载完整权重；`quant` 才使用专用 GPU 和真实量化。
  `modelzoo` 在指定消费方 checkout 与其 SDK 环境中执行。
- 使用确定性输入，保留触发问题的层类型、expert 路由、tied head、shape/dtype 与状态转换。
  不 mock 被测运算，不用零层模型、无条件 skip 或弱化断言缩短门禁。
- fixture 先查 [已有 block](fixtures/README.md)；浮点输入可复用裁剪，已量化 block 用于消费。
  缺失时从原权重取保留机制的最少层，shared-KV 同时保留实际 donor/borrower，
  per-layer 张量同步切分。新 block 先核对真实模型 key/shape 与前向，再做量化；
  不把缩小 hidden/expert 数的合成配置写成原模型 block。源权重不覆盖。
- 保存测试检查配置和真实量化载荷；重载测试调用真实 loader 并验证张量/前向。
  签名、dry-run、文件存在测试分别报告能力边界，不冒充算法精度或部署验证。
- runner/selector 只读取 `--changed-files-file` 或 `CI_TEST_CHANGED_FILES_FILE`，
  不执行 Git、不读取 `.git`。外层负责 checkout、整个 patchset 的路径列表及版本校验；
  列表为空只跑 policy，未提供列表按未知变更跑全集，文件缺失/格式错误必须失败。
  commit 仅记录外部输入，缺失记为未知，不宣称已校验 checkout 或工作区状态。
- 修改 lane/selector/runner 要测已知空 diff、未知 diff、非法路径列表、rename 的两端、共享输入、
  新增/删除用例和失败状态传播。文档也运行 policy；显式错误输入不静默回退。
- 依赖固定测试环境；缺包、缺 GPU、缺模型或错版本直接失败，不能变成绿色 skip。
  checkpoint 下载需 SHA-256；本地模型路径是显式输入，在运行记录中保留。
- 新模型按实际消费方补契约；不要让一个 Qwen smoke 代表所有架构。修改公共格式时显式跑
  下游消费测试，必要的 HMONNX block 导出仍由 ModelZoo CI 负责。
- 量化支持按 [能力矩阵](quantization-matrix.md) 记录算法、数值类型、位宽/分组、结构、格式
  和验证证据。新能力须有保留触发结构的真实量化回归；只新增模型名且完全复用旧机制时，
  可复用算法代表集，同时补本模型模块树/校准/参数等真实功能测试。
