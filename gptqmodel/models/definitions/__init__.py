# isort: off
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Many model architectures inherit from LlamaGPTQ, so it’s necessary to import llama first to avoid circular imports.
from .llama import LlamaQModel

# other model
# from .baichuan import BaiChuanGPTQ
# from .bloom import BloomGPTQ
# from .chatglm import ChatGLM
# from .codegen import CodeGenGPTQ
# from .cohere import CohereGPTQ
# from .cohere2 import Cohere2GPTQ
# from .dbrx import DbrxGPTQ
# from .dbrx_converted import DbrxConvertedGPTQ
# from .decilm import DeciLMGPTQ
# from .deepseek_v2 import DeepSeekV2GPTQ
# from .deepseek_v3 import DeepSeekV3GPTQ
# from .dream import DreamGPTQ
# from .exaone import ExaoneGPTQ
# from .ernie4_5 import ERNIE4_5GPTQ
# from .ernie4_5_moe import ERNIE4_5_MOEGPTQ
# from .gemma import GemmaGPTQ
# from .gemma2 import Gemma2GPTQ
# from .gemma3 import Gemma3GPTQ
# from .glm import GLM
# from .gpt2 import GPT2GPTQ
# from .gpt_bigcode import GPTBigCodeGPTQ
# from .gpt_neox import GPTNeoXGPTQ
# from .gptj import GPTJGPTQ
# from .granite import GraniteGPTQ
# from .grinmoe import GrinMOEGPTQ
# from .hymba import HymbaGPTQ
# from .instella import InstellaGPTQ
# from .internlm import InternLMGPTQ
# from .internlm2 import InternLM2GPTQ
# from .llama import LlamaGPTQ
# from .longllama import LongLlamaGPTQ
# from .mimo import MimoGPTQ
# from .minicpm3 import MiniCPM3GPTQ
# from .mistral import MistralGPTQ
# from .mixtral import MixtralGPTQ
# from .mllama import MLlamaGPTQ
# from .mobilellm import MobileLLMGPTQ
# from .moss import MOSSGPTQ
# from .mpt import MPTGPTQ
# from .olmo2 import Olmo2GPTQ
# from .opt import OPTGPTQ
# from .ovis import OvisGPTQ
# from .phi import PhiGPTQ
# from .phi3 import Phi3GPTQ
# from .qwen import QwenGPTQ
# from .qwen2 import Qwen2GPTQ
# from .qwen2_5_vl import Qwen2_5_VLGPTQ
# from .qwen2_moe import Qwen2MoeGPTQ
# from .qwen2_vl import Qwen2VLGPTQ
# from .qwen3 import Qwen3GPTQ
# from .qwen3_moe import Qwen3MoeGPTQ
# from .gpt_oss import GptOssGPTQ
# from .rw import RWGPTQ
# from .stablelmepoch import StableLMEpochGPTQ
# from .starcoder2 import Starcoder2GPTQ
# from .telechat2 import TeleChat2GPTQ
# from .xverse import XverseGPTQ
# from .yi import YiGPTQ
from .falcon_h1 import FalconH1QModel
# from .pangu_alpha import PanguAlphaGPTQ
from .baichuan import BaiChuanQModel
from .bloom import BloomQModel
from .brumby import BrumbyQModel
from .chatglm import ChatGLMQModel
from .codegen import CodeGenQModel
from .dbrx import DbrxQModel
from .dbrx_converted import DbrxConvertedQModel
from .decilm import DeciLMQModel
from .deepseek_v2 import DeepSeekV2QModel
from .deepseek_v2_ocr import DeepSeekV2OCRQModel
from .deepseek_v3 import DeepSeekV3QModel
from .dots1 import Dots1QModel
from .dream import DreamQModel
from .exaone import ExaOneQModel
from .exaone4 import Exaone4QModel
from .ernie4_5 import Ernie4_5QModel
from .ernie4_5_moe import Ernie4_5_MoeQModel
from .gemma2 import Gemma2QModel
from .gemma3 import Gemma3QModel
from .glm import GlmQModel
from .gpt2 import GPT2QModel
from .gpt_bigcode import GptBigCodeQModel
from .gpt_neo import GptNeoQModel
from .gpt_neox import GPTNeoXQModel
from .gptj import GptJQModel
from .grinmoe import GrinMoeQModel
from .hymba import HymbaQModel
from .instella import InstellaQModel
from .internlm import InternLMQModel
from .internlm2 import InternLM2QModel
from .llama4 import Llama4QModel
from .mimo import MimoQModel
from .minicpm3 import MiniCpm3QModel
from .minimax_m2 import MiniMaxM2GPTQ
from .mixtral import MixtralQModel
from .mllama import MLlamaQModel
from .mobilellm import MobileLLMQModel
from .moss import MossQModel
from .mpt import MptQModel
from .opt import OptQModel
from .ovis import OvisQModel
from .phi import PhiQModel
from .phi3 import Phi3QModel
from .qwen import QwenQModel
from .qwen2 import Qwen2QModel
from .qwen2_5_vl import Qwen2_5_VLQModel
from .qwen2_moe import Qwen2MoeQModel
from .qwen2_vl import Qwen2VLQModel
from .qwen3 import Qwen3QModel
try:
    from .qwen3_5 import Qwen3_5QModel
except ImportError:
    Qwen3_5QModel = None

try:
    from .qwen3_5_moe import Qwen3_5_MoeQModel, Qwen3_5MoeQModel
except ImportError:
    Qwen3_5_MoeQModel = None
    Qwen3_5MoeQModel = None
from .qwen3_moe import Qwen3MoeQModel
from .qwen3_vl import Qwen3_VLQModel
from .rw import RwgQModel
from .starcoder2 import Starcoder2QModel
from .telechat2 import TeleChat2QModel
from .xverse import XverseQModel
from .falcon_h1 import FalconH1QModel
from .pangu_alpha import PanguAlphaQModel
from .longcat_flash import LongCatFlashQModel
from .apertus import ApertusQModel
from .klear import KlearQModel
from .llava_qwen2 import LlavaQwen2QModel
from .nemotron_h import NemotronHQModel
from .qwen3_omni_moe import Qwen3OmniMoeGPTQ
from .mistral3 import Mistral3GPTQ
from .afmoe import AfMoeQModel
from .glm4v import Glm4vGPTQ
from .voxtral import VoxtralGPTQ
