# DeepSeek-V4-Flash 适配周报

本周完成 DeepSeek-V4-Flash 的整体适配，mHC 用融合算子替代 20 轮展开，显著减少导出节点和时间。
SWA 使用 128 滑窗，CSA 使用 4:1 压缩缓存并逐 token Top-512，HCA 使用 128:1 压缩缓存，三者均按固定静态图重新设计。
目前短文本与真实连续中文的 256、1K、2K、4K、8K 阶梯均正常，最终 8K/max512 在第 241 个生成 token 正常 EOS，完成 32 次 prefill、240 次 decode 且无周期退化。
