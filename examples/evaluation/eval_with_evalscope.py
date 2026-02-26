import argparse
from typing import Any, Dict, List, Optional

import torch
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.infer_utils import parse_gpt_oss_output
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.messages import ChatMessage
from evalscope.api.registry import register_model_api
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope import TaskConfig, run_task


@register_model_api(name="EvalModel")
class EvalModel(ModelAPI):

    def __init__(
        self,
        model_name,
        model,
        tokenizer: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.base_url = base_url
        self.api_key = api_key
        self.config = config

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = []
        for message in input:
            messages.append({"role": message.role, "content": message.content})
        if 'gpt_oss' in self.model_name or 'gpt-oss' in self.model_name.lower():
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # gpt_oss模型默认是medium，还可设为low、high
                reasoning_effort=getattr(self.config, 'reasoning_effort', 'medium'),
            )
            # print(f"reasoning_effort: {getattr(self.config, 'reasoning_effort', 'medium')}")
            # print(prompt)
        elif 'qwen3' in self.model_name or 'qwen3' in self.model_name.lower():
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=getattr(self.config, "enable_thinking", False),
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # 对于多GPU模型，获取第一个GPU设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # 对于多GPU模型，使用第一个GPU
            device = list(self.model.hf_device_map.values())[0]
        else:
            device = "cuda:0"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=getattr(self.config, "max_tokens", 2048),
            do_sample=getattr(self.config, "do_sample", False),
            temperature=getattr(self.config, "temperature", 0.0),
            pad_token_id=self.tokenizer.eos_token_id
        )
        if hasattr(self.config, "enable_thinking"):
            generation_kwargs["enable_thinking"] = getattr(self.config, "enable_thinking")
        generated_ids = self.model.generate(**generation_kwargs)[0]
        input_length = inputs["input_ids"].shape[1]
        output_text = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
        response = output_text
        if 'gpt_oss' in self.model_name or 'gpt-oss' in self.model_name.lower():
            result = parse_gpt_oss_output(output_text)
            response = result['final']
        # print(output_text)
        # print(response)
        return ModelOutput.from_content(model=self.model_name, content=response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 EvalScope 评测 GPTQModel/Transformers 模型")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data02/datasets/Qwen3-Coder-30B-A3B-Instruct/",
        help="模型路径",
    )
    parser.add_argument(
        "--model-loader",
        type=str,
        choices=["float", "quantized"],
        default="float",
        help="模型加载方式：float 表示 AutoModelForCausalLM，quantized 表示 GPTQModel.load",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        help="量化模型后端，示例：triton/cuda/auto",
    )
    parser.add_argument(
        "--auto-device-map",
        action="store_true",
        default=True,
        help="多卡时使用 device_map=auto",
    )
    parser.add_argument(
        "--disable-auto-device-map",
        action="store_true",
        help="关闭多卡自动切分，强制按 --device 加载",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        type=str,
        default="70GB",
        help="float 多卡加载时每张 GPU 的最大显存限制",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="是否启用 trust_remote_code",
    )
    parser.add_argument(
        "--disable-trust-remote-code",
        action="store_true",
        help="关闭 trust_remote_code",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        default=True,
        help="是否使用 fast tokenizer",
    )
    parser.add_argument(
        "--disable-fast-tokenizer",
        action="store_true",
        help="关闭 fast tokenizer",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["auto", "gpt_oss", "qwen3", "glm", "generic"],
        default="auto",
        help="模型家族，用于构建 GenerateConfig（glm 用于 GLM-4.7-Flash 等）",
    )
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=1234,
        help="torch.manual_seed",
    )
    args = parser.parse_args()
    if args.disable_auto_device_map:
        args.auto_device_map = False
    if args.disable_trust_remote_code:
        args.trust_remote_code = False
    if args.disable_fast_tokenizer:
        args.use_fast_tokenizer = False
    return args


def get_evalscope_task_config() -> Dict[str, Any]:
    """
    统一配置 EvalScope TaskConfig 相关参数。
    后续如需改动测试集与任务配置，请只修改这个函数。
    """
    return {
        "device": "cuda",
        "max_tokens": 2048,
        "temperature": 0.0,
        "do_sample": False,
        "reasoning_effort": "low",
        "enable_thinking": True,
        "datasets": ["humaneval"],
        "dataset_args": {
            "cmmlu": {
                "subset_list": [
                    "college_actuarial_science",
                    "college_medicine",
                    "college_law",
                    "machine_learning",
                ]
            }
        },
        "dataset_max_tokens": {
            "humaneval": 8192,
        },
        "work_dir": "./outputs-evalscope",
        "timeout": 600,
        "eval_batch_size": 1,
        "seed": 42,
        "limit": None,
    }


def resolve_backend(backend_name: str):
    attr_name = backend_name.upper()
    if not hasattr(BACKEND, attr_name):
        valid_backends = [k for k in dir(BACKEND) if k.isupper()]
        raise ValueError(f"无效 backend: {backend_name}，可选: {valid_backends}")
    return getattr(BACKEND, attr_name)


def detect_model_family(model_path: str, model_family: str) -> str:
    if model_family != "auto":
        return model_family
    lower_path = model_path.lower()
    if "gpt_oss" in lower_path or "gpt-oss" in lower_path:
        return "gpt_oss"
    if "qwen3" in lower_path:
        return "qwen3"
    if "glm" in lower_path:
        return "glm"
    return "generic"


def maybe_apply_glm_convert_for_float_model(model: Any, model_family: str) -> None:
    """仅对 AutoModelForCausalLM 路径尝试 GLM 结构转换。"""
    if model_family != "glm":
        return
    try:
        from gptqmodel.models.definitions.glm4_moe_lite import convert_glm4_moe_lite_convert
        convert_glm4_moe_lite_convert(model, model.config)
        print("已应用 GLM convert_glm4_moe_lite_convert 适配")
    except Exception as e:
        print(f"GLM convert 适配未生效（可能不需要），继续评测。error: {e}")

def load_model_and_tokenizer(args: argparse.Namespace, task_config: Dict[str, Any]):
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 张GPU")
    device = task_config.get("device", "cuda")
    model_family = detect_model_family(args.model_path, args.model_family)

    if args.model_loader == "quantized":
        backend = resolve_backend(args.backend)
        load_kwargs: Dict[str, Any] = {
            "backend": backend,
            "trust_remote_code": args.trust_remote_code,
        }
        if gpu_count > 1 and args.auto_device_map:
            print(f"使用 {gpu_count} 张GPU加载量化模型")
            load_kwargs["device_map"] = "auto"
        else:
            print("使用单卡方式加载量化模型")
            load_kwargs["device"] = device

        model = GPTQModel.load(args.model_path, **load_kwargs)
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer)
    else:
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": args.trust_remote_code,
        }
        if gpu_count > 1 and args.auto_device_map:
            print(f"使用 {gpu_count} 张GPU加载浮点模型")
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "max_memory": {i: args.max_memory_per_gpu for i in range(gpu_count)},
                }
            )
        else:
            print("使用单卡方式加载浮点模型")
            load_kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        maybe_apply_glm_convert_for_float_model(model, model_family)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer)

    print(f"------- load model from {args.model_path} --------")
    return model, tokenizer


def build_generate_config(
    args: argparse.Namespace,
    task_config: Dict[str, Any],
    dataset: str,
    dataset_max_tokens: Dict[str, int],
) -> GenerateConfig:
    family = detect_model_family(args.model_path, args.model_family)
    max_tokens = dataset_max_tokens.get(dataset, task_config.get("max_tokens", 2048))
    temperature = task_config.get("temperature", 0.0)
    do_sample = task_config.get("do_sample", False)
    reasoning_effort = task_config.get("reasoning_effort", "low")
    enable_thinking = task_config.get("enable_thinking", True)

    if family == "gpt_oss":
        return GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            reasoning_effort=reasoning_effort,
        )

    if family == "qwen3":
        return GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            enable_thinking=enable_thinking,
        )

    return GenerateConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )


def run_evaluation(args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
    task_config = get_evalscope_task_config()
    datasets: List[str] = task_config.get("datasets", [])
    dataset_args: Dict[str, Dict[str, List[str]]] = task_config.get("dataset_args", {})
    dataset_max_tokens: Dict[str, int] = task_config.get("dataset_max_tokens", {})
    work_dir: str = task_config.get("work_dir", "./outputs-evalscope")
    timeout: int = task_config.get("timeout", 600)
    eval_batch_size: int = task_config.get("eval_batch_size", 1)
    seed: int = task_config.get("seed", 42)
    limit = task_config.get("limit", None)

    if not datasets:
        raise ValueError("get_eval_dataset_config() 中 datasets 为空，请在该函数中配置至少一个测试集")

    for dataset in datasets:
        print(f"---------- dataset: {dataset} ----------")
        generate_config = build_generate_config(args, task_config, dataset, dataset_max_tokens)
        print(
            "generate_config:",
            f"max_tokens={getattr(generate_config, 'max_tokens', None)}",
            f"temperature={getattr(generate_config, 'temperature', None)}",
            f"do_sample={getattr(generate_config, 'do_sample', None)}",
            f"reasoning_effort={getattr(generate_config, 'reasoning_effort', None)}",
            f"enable_thinking={getattr(generate_config, 'enable_thinking', None)}",
        )

        evalscope_model = EvalModel(
            model_name=args.model_path,
            model=model,
            tokenizer=tokenizer,
            config=generate_config,
        )

        try:
            task_cfg_kwargs: Dict[str, Any] = {
                "model": evalscope_model,
                "datasets": [dataset],
                "dataset_args": dataset_args,
                "work_dir": work_dir,
                "timeout": timeout,
                "eval_batch_size": eval_batch_size,
                "seed": seed,
            }
            if limit is not None:
                task_cfg_kwargs["limit"] = limit
            task_cfg = TaskConfig(**task_cfg_kwargs)
            run_task(task_cfg=task_cfg)
        except Exception as error:
            print(f"---------- dataset: {dataset} ---------- eval failed, error: {error}")


if __name__ == "__main__":
    """
    评测集：https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html
    
    单GPU运行示例：
    CUDA_VISIBLE_DEVICES=3 python eval_with_evalscope.py  2>&1 | tee logs/evalscope-gpt-oss-20b-4bit-128g-wiki_calib.log
    CUDA_VISIBLE_DEVICES=2 python eval_with_evalscope.py  2>&1 | tee logs/evalscope-gpt-oss-20b-4bit-128g-wiki_calib-hessian_mse.log
    CUDA_VISIBLE_DEVICES=3 python eval_with_evalscope.py  2>&1 | tee logs/evalscope-gpt-oss-20b-origin-reasoning_low.log
    
    多GPU运行120B模型示例：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_with_evalscope.py  2>&1 | tee logs/evalscope-gpt-oss-120b-multi-gpu.log
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_with_evalscope.py  2>&1 | tee logs/evalscope-gpt-oss-120b-8gpu.log
    """
    args = parse_args()
    torch.manual_seed(args.torch_seed)
    task_config = get_evalscope_task_config()
    model, tokenizer = load_model_and_tokenizer(args, task_config)
    run_evaluation(args, model, tokenizer)