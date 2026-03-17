from .rotation import *

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.definitions.qwen3_next import Qwen3NextGPTQ
from gptqmodel.models.definitions.glm4_moe_lite import Glm4MoeLiteQModel
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.models.definitions.deepseek_v2_ocr import DeepSeekV2OCRQModel
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel

MODEL_ROTATION_REGISTRY = [
    {
        "cls": Qwen3QModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms,
        "rotate_fn": rotate_model_qwen3_5,
    },
    {
        "cls": Qwen2QModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms,
        "rotate_fn": rotate_model,
    },
    {
        "cls": Qwen3MoeQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_qwen3_5_moe,
        "rotate_fn": rotate_model_qwen3_5_moe,
    },
    {
        "cls": Qwen3_5QModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_qwen3_5,
        "rotate_fn": rotate_model_qwen3_5,
    },
    {
        "cls": Qwen3_5MoeQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_qwen3_5_moe,
        "rotate_fn": rotate_model_qwen3_5_moe,
    },
    {
        "cls": Qwen3NextGPTQ,
        "vision": False,
        "fuse_norm": fuse_layer_norms_qwen3_next,
        "rotate_fn": rotate_model_qwen3_next,
    },
    {
        "cls": Qwen2_5_VLQModel,
        "vision": True,
        "fuse_norm": fuse_layer_norms,
        "fuse_vision_norm": fuse_vision_layer_norms_qwen2_5_vl,
        "rotate_fn": rotate_model_qwen2_5_vl,
        "rotate_vision_fn": rotate_vision_model,
    },
    {
        "cls": Qwen3_VLQModel,
        "vision": True,
        "fuse_norm": fuse_layer_norms,
        "fuse_vision_norm": fuse_vision_layer_norms_qwen3_vl,
        "rotate_fn": rotate_model_qwen3_vl,
        "rotate_vision_fn": rotate_vision_model_qwen3_vl,
    },
    {
        "cls": Glm4MoeLiteQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_deepseekv3_mla_moe,
        "rotate_fn": rotate_model_deepseekv3_mla_moe,
    },
    {
        "cls": DeepSeekV2QModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_deepseekv2_mla_moe,
        "rotate_fn": rotate_model_deepseekv2_mla_moe,
    },
    {
        "cls": DeepSeekV2OCRQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_deepseekv2_ocr,
        "rotate_fn": rotate_model_deepseekv2_ocr,
    },
    {
        "cls": DeepSeekV3QModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_deepseekv3_mla_moe,
        "rotate_fn": rotate_model_deepseekv3_mla_moe,
    },
    {
        "cls": LlamaQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms,
        "rotate_fn": rotate_model,
    },
]


def get_rotation_entry(model):
    for entry in MODEL_ROTATION_REGISTRY:
        if isinstance(model, entry["cls"]):
            return entry
    return None
