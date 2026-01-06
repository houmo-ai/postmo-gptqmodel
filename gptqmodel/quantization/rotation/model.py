from .rotation import *

def get_rotation_entry(model):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.models.definitions.qwen2 import Qwen2QModel
    from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
    from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel

    MODEL_ROTATION_REGISTRY = [
        {
            "cls": LlamaQModel,
            "vision": False,
            "rotate_fn": rotate_model,
        },
        {
            "cls": Qwen2QModel,
            "vision": False,
            "rotate_fn": rotate_model,
        },
        {
            "cls": Qwen2_5_VLQModel,
            "vision": True,
            "rotate_fn": rotate_model_qwen3_vl,
            "rotate_vision_fn": rotate_vision_model,
        },
        {
            "cls": Qwen3_VLQModel,
            "vision": True,
            "rotate_fn": rotate_model_qwen3_vl,
            "rotate_vision_fn": rotate_vision_model,
        },
    ]

    for entry in MODEL_ROTATION_REGISTRY:
        if isinstance(model, entry["cls"]):
            return entry
    return None
