帮我完成 Gemma4-E2B 模型的的 autoround 适配和量化, 这个模型对应的权重文件在 /data01/datasets/gemma-4-E2B


你需要：
* 完成 autoround 对 Gemma4-E2B 的适配
* 量化并导出 GPTQModel 的测试，前期可以做烟测(少量的校准集和 iters),验证量化后能正常输出后，就进行完整的量化实验. 你需要同时用 8bit 和 4bit 量化
* 评估 量化前后模型的 ppl( 参考scripts_gemma4/eval/eval_ppl.py),  mmlu_pro, ceval,humaneval( 这三个使用 evalscope)
* 对浮点模型评估对应的数据集精度
* 汇总量化前后的结果,汇总为md 表格给我