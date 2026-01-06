import argparse
from pathlib import Path
import os
import torch
import json  
from datasets import load_dataset
from gptqmodel.utils.eval import EVAL
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from safetensors.torch import load_file as load_safetensors_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/data01/nfs_shared/llm_models/Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--quarot_model_path", type=str, default=None)
    parser.add_argument("--quant_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--with_random", action="store_true", default=False)
    parser.add_argument("--rotation", type=str, default=None, choices=['random', 'hadamard'])
    parser.add_argument("--hessian_mse", action="store_true", default=False)
    # parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()

def create_template_messages(image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages

def format_qwen2_vl_dataset(image, text, assistant):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": assistant},
    ]

def format_qwen2_vl_dataset_for_calibration(item):
    content = []
    for p_input in item["struct"]:
        content.append({"type": p_input["type"], p_input["type"]: p_input["value"]})
    format_item = [{"role": "user", "content": content}, {"role": "assistant", "content": item["response"]}]
    return format_item


def main():
    args = get_args()

    if args.quant_path and Path(args.quant_path).exists():
        # model = GPTQModel.load(args.quant_path, attn_implementation="flash_attention_2")
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.quant_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")
        # model.model.to(torch.half)
        processor = AutoProcessor.from_pretrained(args.quant_path)

        for image_path in os.listdir(args.image_dir):   
            # check image to text
            if image_path != "a.png":
                continue
            messages = create_template_messages(
                os.path.join(args.image_dir, image_path),
                "请根据这张图写一篇800字左右的作文",
            )
    
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs = Qwen2VLQModel.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            # if not args.with_random:
            #     generated_ids = model.generate(**inputs, max_new_tokens=2048, top_p=1, top_k=1, temperature=1)
            # else:
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            # generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print("image_path:", image_path, "output_text:", output_text)
            breakpoint()
    else:
        quant_config = QuantizeConfig(bits=4, group_size=64, sym=True, damp_percent=0.01, mse=2.4, desc_act=False, rotation=args.rotation, rotation_vision=args.rotation, offload_to_disk=False, hessian_mse=args.hessian_mse)
        # quant_config = QuantizeConfig(bits=4, group_size=64, sym=True, damp_percent=0.01, mse=2.4, desc_act=False)
        # model = GPTQModel.load(args.model_id, quant_config, dtype=torch.float16, attn_implementation="flash_attention_2")
        model = GPTQModel.load(args.model_id, quant_config)
        breakpoint()
        if args.quarot_model_path:
            state_dict = load_safetensors_file(args.quarot_model_path)
            model.model.load_state_dict(state_dict)
        calibration_dataset = json.load(open("data/calib_data/result_base_case_qwen2_5_vl.json", "r"))
        calibration_dataset = [format_qwen2_vl_dataset(item["struct"][0]["content"][0]["image"], item["struct"][0]["content"][1]["text"], item["response"]) for item in calibration_dataset]
        
        cmmmu_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json", "r"))
        cmmmu_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in cmmmu_val_struct]

        cmmmu_val_struct = sorted(cmmmu_val_struct, key=lambda x: len(x[1]["content"]))[-128:]
        calibration_dataset = calibration_dataset + cmmmu_val_struct

        mmmu_dev_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json", "r"))
        mmmu_dev_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in mmmu_dev_val_struct]
        mmmu_dev_val_struct = sorted(mmmu_dev_val_struct, key=lambda x: len(x[1]["content"]))[-64:]
        calibration_dataset = calibration_dataset + mmmu_dev_val_struct

        ocrbench_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json", "r"))
        ocrbench_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in ocrbench_struct]
        ocrbench_struct = sorted(ocrbench_struct, key=lambda x: len(x[1]["content"]))[-32:]
        calibration_dataset = calibration_dataset + ocrbench_struct

        docvqa_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json", "r")) 
        docvqa_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in docvqa_val_struct]
        docvqa_val_struct = sorted(docvqa_val_struct, key=lambda x: len(x[1]["content"]))[-32:]
        calibration_dataset = calibration_dataset + docvqa_val_struct

        coco_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json", "r"))
        coco_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in coco_val_struct]
        coco_val_struct = sorted(coco_val_struct, key=lambda x: len(x[1]["content"]))[-32:]
        calibration_dataset = calibration_dataset + coco_val_struct

        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.quant_path)



if __name__ == "__main__":
    main()
