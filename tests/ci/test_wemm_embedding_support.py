from __future__ import annotations

import json

from gptqmodel.models.auto import MODEL_MAP, check_and_get_model_definition
from gptqmodel.models.definitions.wemm_embedding import WeMMEmbeddingQModel


def test_wemm_embedding_definition_is_registered():
    assert MODEL_MAP["wemm_embedding"] is WeMMEmbeddingQModel


def test_wemm_embedding_auto_map_selects_qwen35_vlm_definition(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "auto_map": {
                    "AutoModel": "modeling_wemm_embedding.WeMMEmbedding",
                    "AutoModelForCausalLM": "modeling_wemm_embedding.WeMMEmbedding",
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        check_and_get_model_definition(str(tmp_path), trust_remote_code=True)
        is WeMMEmbeddingQModel
    )
