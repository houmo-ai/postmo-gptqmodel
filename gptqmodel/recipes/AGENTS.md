# 对外量化 recipe

- recipe 是 ModelZoo 等消费方的公开入口。命名参数、默认值、返回对象的 `output_dir`
  和 provenance 是接口的一部分；新增参数须贯通校验、命令/后端执行与测试。
- 不静默丢弃未知参数，不把 `**kwargs` 过滤当兼容修复。只记录而不执行的兼容字段必须
  明确说明；dry-run 生成了命令不代表参数已经在算法中生效。
- recipe 负责量化编排和执行记录；通用求解、打包、HF 保存能力放回所属模块。
  不引入 HMONNX 导出、ModelZoo workflow 配置或特定机器绝对路径作为隐式依赖。
- 校准输入、样本数、长度、seed、格式和精度应可追溯。失败执行不可发布成功产物记录。
  源 checkpoint 不原地覆盖；复用已有 packed 权重时校验实际配置与张量布局。
- 改接口时一起核对 `examples/quantization/`、随仓 AutoRound parser、生产端 API 回归，
  以及 ModelZoo quant adapter。重命名或删除参数必须提供消费方联调结果。
- 新 recipe 同一变更提供可运行用例和同名 `.impact.json`，不修改其他模型的测试声明；
  改公共能力仍需回归旧使用方。最小模型需保留相关模态、attention、
  expert 或 head 结构。契约检查与真实量化分开，规则见 [CI 说明](../../tests/ci/README.md)。
