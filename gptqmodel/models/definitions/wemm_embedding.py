# SPDX-License-Identifier: Apache-2.0
"""GPTQModel definition for Tencent WeMM-Embedding checkpoints.

WeMM-Embedding is a Qwen3.5 multimodal checkpoint with a small remote-code
pooling helper.  Quantization uses the maintained Qwen3.5 VLM module tree;
this named subclass makes the model family explicit for callers that inspect
the definition rather than only ``config.model_type``.
"""

from .qwen3_5_vl import Qwen3_5_VLQModel


class WeMMEmbeddingQModel(Qwen3_5_VLQModel):
    """Qwen3.5 VLM GPTQ definition for WeMM-Embedding."""


__all__ = ["WeMMEmbeddingQModel"]
