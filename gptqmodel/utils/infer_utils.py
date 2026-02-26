from transformers import AutoTokenizer
from gptqmodel import QuantizeConfig, BACKEND, GPTQModel


def load_quantized_model(model_path, device, backend=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("✅ Tokenizer loaded.")

    # Load quantized model
    print(f"⚙️  Loading GPTQ model from {model_path} on {device} (backend={backend})")
    model = GPTQModel.from_quantized(model_path, device=device, backend=backend)
    # model = GPTQModel.load(model_path, device=device, backend=backend)
    model.eval()
    print("✅ Model loaded.")
    return model, tokenizer


def parse_gpt_oss_output(output):
    """
    解析GPT-OSS模型的特定格式输出
    
    output样式：<|channel|>analysis<|message|> ... <|channel|>commentary<|message|> ... <|channel|>final<|message|> ... <|end|>
    """
    result = {
        "analysis": "",
        "commentary": "",
        "final": ""
    }
    
    # 定义频道标记
    channel_markers = ["<|channel|>analysis<|message|>", 
                      "<|channel|>commentary<|message|>", 
                      "<|channel|>final<|message|>"]
    
    # 提取analysis部分
    analysis_start_marker = "<|channel|>analysis<|message|>"
    if analysis_start_marker in output:
        analysis_start = output.find(analysis_start_marker) + len(analysis_start_marker)
        # 找下一个标记或结尾
        next_positions = []
        for marker in ["<|channel|>commentary<|message|>", "<|channel|>final<|message|>", "<|end|>"]:
            pos = output.find(marker, analysis_start)
            if pos != -1:
                next_positions.append(pos)
        
        analysis_end = min(next_positions) if next_positions else len(output)
        result["analysis"] = output[analysis_start:analysis_end].strip()
    
    # 提取commentary部分
    commentary_start_marker = "<|channel|>commentary<|message|>"
    if commentary_start_marker in output:
        commentary_start = output.find(commentary_start_marker) + len(commentary_start_marker)
        # 找下一个标记或结尾
        next_positions = []
        for marker in ["<|channel|>analysis<|message|>", "<|channel|>final<|message|>", "<|end|>"]:
            pos = output.find(marker, commentary_start)
            if pos != -1:
                next_positions.append(pos)
        
        commentary_end = min(next_positions) if next_positions else len(output)
        result["commentary"] = output[commentary_start:commentary_end].strip()
    
    # 提取final部分
    final_start_marker = "<|channel|>final<|message|>"
    if final_start_marker in output:
        final_start = output.find(final_start_marker) + len(final_start_marker)
        # 找到结束标记或结尾
        end_marker_pos = output.find("<|end|>", final_start)
        if end_marker_pos != -1:
            result["final"] = output[final_start:end_marker_pos].strip()
        else:
            result["final"] = output[final_start:].strip()
    
    # 如果没有找到明确的格式，将整个输出作为final
    if not any(result.values()):
        # 移除特殊标记
        clean_output = output
        for marker in ["<|start|>", "<|end|>", "<|message|>", "<|channel|>"]:
            clean_output = clean_output.replace(marker, "")
        result["final"] = clean_output.strip()
    
    return result

if __name__ == "__main__":
    # 使用示例
    output_tokens = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-09-10\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions\n\n你是一个专业的AI助手，请用中文回答问题。<|end|><|start|>user<|message|>请介绍一下人工智能的发展历史<|end|><|start|>assistant<|channel|>analysis<|message|>We need to answer in Chinese, describing the history of AI. Provide a concise but comprehensive overview: early concepts, Turing, 1950s, Dartmouth, early programs, 1960s-70s, 1980s expert systems, 1990s machine learning, 2000s deep learning, 2010s AI boom, 2020s. Mention key milestones: Turing test, ELIZA, perceptron, backprop, etc. Provide timeline. Also mention AI winter. Provide Chinese answer.<|end|><|start|>assistant<|channel|>final<|message|>**人工智能（Artificial Intelligence, AI）发展史（简要概览）**\n\n| 年代 | 重要事件 | 代表性成果 | 关键人物 |\n|------|----------|-----------|----------|\n| **1940s–1950s** | 计算机与逻辑的结合 | 1. 1943–1944：沃伦·麦卡洛克与沃尔特·皮茨提出"

    parsed_result = parse_gpt_oss_output(output_tokens)

    # 打印解析结果
    if parsed_result["analysis"]:
        print("[思维链分析 (analysis)]:")
        print(parsed_result["analysis"])
        print("-" * 30)

    if parsed_result["commentary"]:
        print("[工具调用信息 (commentary)]:")
        print(parsed_result["commentary"])
        print("-" * 30)

    if parsed_result["final"]:
        print("[最终回答 (final)]:")
        print(parsed_result["final"])