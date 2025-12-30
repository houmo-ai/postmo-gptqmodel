# from datasets import load_dataset
# from gptqmodel import GPTQModel, QuantizeConfig

# model_id = "/data01/datasets/Qwen3-8B"
# quant_path = "/data01/datasets/Qwen3-8B-int4"

# calibration_dataset = load_dataset(
#     "allenai/c4",
#     data_files="en/c4-train.00001-of-01024.json.gz",
#     split="train"
#   ).select(range(1024))["text"]

# quant_config = QuantizeConfig(bits=4, group_size=128)

# model = GPTQModel.load(model_id, quant_config)

# # increase `batch_size` to match gpu/vram specs to speed up quantization
# model.quantize(calibration_dataset, batch_size=1)

# model.save(quant_path)

#export TOKENIZERS_PARALLELISM=false
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "/data01/datasets/Qwen3-8B"
quant_path = "/data01/datasets/Qwen3-8B-int4"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calibration_dataset = [
    text for text in dataset["text"] 
    if text.strip() and len(text.strip()) > 50  
][:1024]
print(f"校准数据集样本数：{len(calibration_dataset)}")

quant_config = QuantizeConfig(bits=4, group_size=64)
model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=128)

# 保存量化后的模型
model.save(quant_path)
print(f"量化完成！模型已保存至：{quant_path}")