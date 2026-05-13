from .rotation import *

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5MoeQModel, Qwen3_5MoeTextQModel
from gptqmodel.models.definitions.qwen3_5_vl import Qwen3_5_VLQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
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
        "cls": Qwen3OmniMoeGPTQ,
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
        # qwen3_5_moe is the full VLM model_type; rotate vision with the same
        # LLM Q. qwen3_5_moe_text is registered separately below with no vision.
        "vision": True,
        "fuse_norm": fuse_layer_norms_qwen3_5_moe,
        "fuse_vision_norm": fuse_vision_layer_norms_qwen3_5_vl,
        "rotate_fn": rotate_model_qwen3_5_moe,
        "rotate_vision_fn": rotate_vision_model_qwen3_5_vl,
    },
    {
        "cls": Qwen3_5MoeTextQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms_qwen3_5_moe,
        "rotate_fn": rotate_model_qwen3_5_moe,
    },
    {
        # CRITICAL: Qwen3_5_VLQModel was previously NOT registered. auto.py maps
        # config.model_type=="qwen3_5" to this class, so without this entry
        # apply_rotation() raised "rotation only supports: ...". This is the
        # primary bug behind why example_qwen35dense.py used a dense LlamaQModel
        # path (dropping mtp/visual) and example_qwen35_vl_rotate_fp.py had to
        # hand-call rotation helpers.
        "cls": Qwen3_5_VLQModel,
        "vision": True,
        "fuse_norm": fuse_layer_norms_qwen3_5,
        "fuse_vision_norm": fuse_vision_layer_norms_qwen3_5_vl,
        "rotate_fn": rotate_model_qwen3_5_vl,
        "rotate_vision_fn": rotate_vision_model_qwen3_5_vl,
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
