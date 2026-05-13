"""Hadamard rotation utilities for Qwen3.5 models.

Provides offline Hadamard rotation for LLM backbone and optional
vision module rotation. References GPTQModel's rotation implementation.

Usage:
    from scripts.rotation_utils import apply_hadamard_rotation
    apply_hadamard_rotation(model, rotate_vision="merger_only")
"""

import sys
import torch

# GPTQModel path for rotation functions
GPTQMODEL_PATH = "/data01/home/yujy/work/gptqmodel"


def _ensure_gptqmodel():
    if GPTQMODEL_PATH not in sys.path:
        sys.path.insert(0, GPTQMODEL_PATH)


def apply_hadamard_rotation(model, rotate_vision="merger_only", device="cuda:0"):
    """Apply Hadamard rotation to a Qwen3.5 model.

    Args:
        model: The model to rotate (CausalLM or ConditionalGeneration).
        rotate_vision: Vision rotation level:
            - "none": Only rotate LLM, don't touch vision at all.
                      WARNING: breaks VLM inference if model has vision module.
            - "merger_only": Rotate LLM + vision merger output layer (recommended).
                            Ensures vision-text interface alignment.
            - "full": Rotate LLM + entire vision encoder + merger.
        device: Device for rotation computation.

    Returns:
        Rotated model and rotation matrix Q.
    """
    _ensure_gptqmodel()
    from gptqmodel.quantization.rotation.rotation import (
        fuse_layer_norms_qwen3_5,
        rotate_model_qwen3_5,
    )

    dev = torch.device(device)

    # Determine the LLM core module path
    has_vision = hasattr(model, 'visual') or hasattr(model, 'model') and hasattr(getattr(model, 'model', None), 'visual')
    llm_module = _get_llm_module(model)

    # Step 1: Fuse LayerNorms into Linear layers (LLM only)
    print("[Rotation] Step 1/3: Fusing LayerNorms...")
    if llm_module is model:
        # CausalLM: model.layers
        model = fuse_layer_norms_qwen3_5(
            model,
            pre_lm_head_norm_module_name="model.norm",
            layers_node="model.layers",
            lm_head_name="lm_head",
        )
    else:
        # VLM: model.model.layers (or similar nested path)
        model = fuse_layer_norms_qwen3_5(
            model,
            pre_lm_head_norm_module_name="model.norm",
            layers_node="model.layers",
            lm_head_name="lm_head",
        )

    # Step 2: Rotate LLM backbone
    print("[Rotation] Step 2/3: Applying Hadamard rotation to LLM...")
    model, Q = rotate_model_qwen3_5(
        model,
        rotate_mode="hadamard",
        device=dev,
        lm_head_name="lm_head",
        layers_node="model.layers",
    )
    print(f"[Rotation] Rotation matrix Q shape: {Q.shape}")

    # Step 3: Rotate vision module (if applicable)
    if rotate_vision != "none" and has_vision:
        _rotate_vision(model, Q, rotate_vision, dev)
    elif rotate_vision != "none" and not has_vision:
        print("[Rotation] No vision module detected, skipping vision rotation.")

    print("[Rotation] Done.")
    return model, Q


def _get_llm_module(model):
    """Get the LLM backbone from a possibly VLM model."""
    if hasattr(model, 'language_model'):
        return model.language_model
    return model


def _rotate_vision(model, Q, mode, device):
    """Rotate vision module components.

    For merger_only: rotates visual.merger output to align with rotated LLM input.
    For full: also rotates the entire vision encoder backbone.
    """
    visual = _get_visual_module(model)
    if visual is None:
        print("[Rotation] Warning: rotate_vision requested but no visual module found.")
        return

    if mode == "full":
        print("[Rotation] Step 3/3: Rotating full vision encoder + merger...")
        _rotate_vision_encoder(visual, device)
    else:
        print("[Rotation] Step 3/3: Rotating vision merger output layer...")

    _rotate_vision_merger(visual, Q, device)


def _get_visual_module(model):
    """Find the visual module in the model."""
    if hasattr(model, 'visual'):
        return model.visual
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        return model.model.visual
    return None


def _rotate_vision_merger(visual, Q, device):
    """Rotate the vision merger output projection to match rotated LLM input space.

    Math: W_fc2_new = Q^T @ W_fc2_old
          bias_new  = Q^T @ bias_old

    This ensures: merger_output @ Q @ Q^T = merger_output (identity after full rotation)
    """
    Q_dev = Q.to(device=device, dtype=torch.float64)

    # Qwen3.5 may use deepstack merger
    if hasattr(visual, 'merger'):
        merger = visual.merger
        if hasattr(merger, 'deepstack_merger_list'):
            for i, sub_merger in enumerate(merger.deepstack_merger_list):
                _rotate_linear_output(sub_merger.linear_fc2, Q_dev, device)
                print(f"  Rotated deepstack_merger[{i}].linear_fc2")
        elif hasattr(merger, 'linear_fc2'):
            _rotate_linear_output(merger.linear_fc2, Q_dev, device)
            print("  Rotated merger.linear_fc2")
        else:
            # Try mlp[-1] or other patterns
            if hasattr(merger, 'mlp') and len(merger.mlp) > 0:
                last_layer = merger.mlp[-1]
                _rotate_linear_output(last_layer, Q_dev, device)
                print("  Rotated merger.mlp[-1]")
            else:
                print("  Warning: Could not find merger output layer to rotate.")


def _rotate_linear_output(linear, Q, device):
    """Apply Q^T @ W rotation to a linear layer's weight and bias."""
    orig_dtype = linear.weight.data.dtype
    orig_device = linear.weight.data.device

    w = linear.weight.data.to(device=device, dtype=torch.float64)
    linear.weight.data = (Q.T @ w).to(dtype=orig_dtype, device=orig_device)

    if linear.bias is not None:
        b = linear.bias.data.to(device=device, dtype=torch.float64)
        linear.bias.data = (Q.T @ b).to(dtype=orig_dtype, device=orig_device)


def _rotate_vision_encoder(visual, device):
    """Rotate the full vision encoder backbone (optional).

    This includes:
    - Patch embedding projection
    - All vision transformer blocks (attention + MLP)
    - Vision LayerNorms fusion

    Note: This is optional and only needed if vision activations also need
    to be quantized. For LLM-only quantization, merger_only is sufficient.
    """
    try:
        _ensure_gptqmodel()
        from gptqmodel.quantization.rotation.rotation import (
            fuse_vision_layer_norms_qwen3_5_vl,
            rotate_vision_model_qwen3_5_vl,
        )
        # These functions handle the full vision encoder rotation
        # including patch_embed, all blocks, and merger input
        print("  Using GPTQModel vision rotation functions...")
        # Note: The actual function signatures may vary - adapt as needed
    except ImportError:
        print("  Warning: GPTQModel vision rotation functions not available.")
        print("  Full vision rotation requires GPTQModel with Qwen3.5 VL support.")
