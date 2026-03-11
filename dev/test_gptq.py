from gptqmodel import GPTQModel
from transformers import AutoTokenizer
import torch
torch.set_grad_enabled(False)

def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3.5 chat templates may emit a long reasoning preamble unless
        # thinking mode is disabled at template rendering time.
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)

tokenizer = AutoTokenizer.from_pretrained("/data02/chenzx/ssfp_weight-w4-moequant-20260307")

model = GPTQModel.load("/data02/chenzx/ssfp_weight-w4-moequant-20260307",device="cpu")
pass