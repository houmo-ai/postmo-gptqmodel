#export TOKENIZERS_PARALLELISM=false
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "/data01/datasets/gpt-oss-20b"
quant_path = "/data01/datasets/gpt-oss-20b-int4"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calibration_dataset = [
    text for text in dataset["text"] 
    if text.strip() and len(text.strip()) > 50  
][:128]
print(f"校准数据集样本数：{len(calibration_dataset)}")

quant_config = QuantizeConfig(bits=4, group_size=64)
model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=128)

# 保存量化后的模型
model.save(quant_path)
print(f"量化完成！模型已保存至：{quant_path}")