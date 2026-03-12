from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.mix_precision import run_mix_precision_pipeline

model_id = "/data01/datasets/Qwen3-8B"
mix_precision_output = "/data01/datasets/Qwen3-8B-mix-int4/mix_precision_search.json"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calibration_dataset = [
    text for text in dataset["text"]
    if text.strip() and len(text.strip()) > 50
][:128]
print(f"校准数据集样本数：{len(calibration_dataset)}")

quant_config = QuantizeConfig(
    bits=4,
    group_size=64,
    mix_precision_search=True,
)

# 覆盖混合精度搜索配置（会先搜索，再把结果写入 quant_config.dynamic）
quant_config.extension_set("mix_precision", {
    "enabled": True,
    "weight_bits": [4, 8],
    "policy": "topk",
    "topk": 0.1,
    "metric": "kl",
    "output_path": mix_precision_output,
})

model = GPTQModel.load(model_id, quant_config)

# 只做混合精度搜索，不执行正式量化
allocation = run_mix_precision_pipeline(
    model=model.model,
    quantize_config=quant_config,
    calibration_data=calibration_dataset,
    device=quant_config.device,
    mix_precision_config_template=model.mix_precision_config_template(),
)

dynamic = quant_config.dynamic or {}
print(f"混合精度搜索完成，dynamic 覆盖层数：{len(dynamic)}")
if allocation is not None:
    print(f"候选/分配层数：{len(allocation)}")
    sample_items = list(dynamic.items())[:10]
    print("dynamic 示例（前10项）:")
    for name, cfg in sample_items:
        print(f"  {name}: {cfg}")

print(f"混合精度搜索结果已保存至：{mix_precision_output}")
