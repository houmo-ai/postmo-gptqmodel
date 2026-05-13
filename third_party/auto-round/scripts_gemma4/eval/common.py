import os
import sys


def ensure_local_gptqmodel_path(root: str | None = None) -> str | None:
    candidate = os.path.expanduser(root or "~/gptqmodel")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
    if candidate in sys.path:
        return candidate
    return None


def patch_gptqmodel_for_gemma4() -> bool:
    ensure_local_gptqmodel_path()
    try:
        from gptqmodel.models import auto as auto_models
        from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
    except Exception:
        return False

    auto_models.MODEL_MAP["gemma4"] = Gemma3ForConditionalGenerationGPTQ
    if "gemma4" not in auto_models.SUPPORTED_MODELS:
        auto_models.SUPPORTED_MODELS.append("gemma4")
    return True
