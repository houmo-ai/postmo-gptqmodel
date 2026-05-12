#export TOKENIZERS_PARALLELISM=false
#ppl:gptqmodel/examples/benchmark/perplexity.py

from gptqmodel import GPTQModel, QuantizeConfig

model_id = "/data02/datasets/Qwen3-30B-A3B"
quant_path = "/data01/datasets/Qwen3-MOE-4bit"

# --------------- quantization with EBSS self-generated calibration ---------------
# This follows the same QuantizeConfig-driven integration used by
# examples/quantization/examples/example_glm_4_7_flash.py.
quant_config = QuantizeConfig(
    bits=4,
    group_size=64,
    rotation="random",
    offload_to_disk=False,
    calibration_generation={
        "enabled": True,
        "provider": "moe_ebss",
        "output_path": "work_dirs/ebss_qwen3_moe_calib.jsonl",
        "nsamples": 1024,
        "seqlen": 512,
        "num_beams": 2,
        "seed": 42,
        "overwrite": True,
        "allow_override": True,
    },
)

model = GPTQModel.load(model_id, quant_config)
# calibration=None means quantize() will resolve the source through
# QuantizeConfig.calibration_generation -> MoeEbssProvider.generate().
model.quantize(calibration=None, batch_size=1)

model.save(quant_path)
print(f"量化完成！模型已保存至：{quant_path}")