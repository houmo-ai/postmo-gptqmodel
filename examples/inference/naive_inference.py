import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data02/datasets/GLM-4.7-Flash"
messages = [{"role": "user", "content": "你多大了？用中文回答。"}]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
from gptqmodel.models.definitions.glm4_moe_lite import convert_glm4_moe_lite_mla_convert
convert_glm4_moe_lite_mla_convert(model,model.config)
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
output_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:])
print(output_text)