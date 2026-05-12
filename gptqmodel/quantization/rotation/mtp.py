# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 MTP (multi-token-prediction) tensor-level rotation post-processor.

transformers strips `mtp.*` from the model on load (see
`_keys_to_ignore_on_load_unexpected`), so by the time GPTQModel finishes
quantizing the LLM, the in-memory model never had an MTP module to rotate.

This module operates directly on the source `safetensors` files: it reads the
original (un-fused, un-rotated) `mtp.*`, `lm_head.weight` and `model.norm.weight`
tensors from the source model directory, applies the same fuse-norms +
rotate(Q_block) recipe that GPTQModel applied to the LLM, and writes the result
to a single sidecar file `mtp_rotated.safetensors` inside the save directory,
also patching `model.safetensors.index.json` so the new tensors are discoverable.

Two artifacts are produced for every MTP block:
  1. mtp.* mirrors of the original tensors, fused & rotated to live in the
     LLM's rotated hidden-state space.
  2. mtp.lm_head_weight — a NEW tensor (shape == original lm_head.weight,
     dtype == lm_head.dtype). This is the dedicated MTP head, fused with
     mtp.norm and rotated with the LLM Q_block. dflash-side inference must
     prefer this buffer over the shared lm_head when present.
  3. mtp.language_model_norm_scale / mtp.mtp_norm_scale — the original
     `(1 + weight)` final-norm multipliers. Rotated checkpoints reset those
     norms after fusion, but MTP consumes norm outputs from both LLM and MTP
     sources; runtimes need these scales to restore the correct source-specific
     hidden state before feeding the next MTP step.
     The sidecar also stores the mathematically equivalent rotated-basis
     transforms `Q.T @ diag(scale) @ Q`, because hidden states in a rotated
     checkpoint are already in the Q basis and cannot be corrected by an
     elementwise multiply in the original basis.

This processor is invariant to the LLM rotate function used and only requires
the final Q_block returned by `rotate_model_qwen3_5(_moe)`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ...utils.logger import setup_logger

log = setup_logger()


MTP_SIDECAR_FILENAME = "mtp_rotated.safetensors"
MTP_LANGUAGE_MODEL_NORM_SCALE_KEY = "mtp.language_model_norm_scale"
MTP_MTP_NORM_SCALE_KEY = "mtp.mtp_norm_scale"
MTP_LANGUAGE_MODEL_NORM_ROTATED_MATRIX_KEY = "mtp.language_model_norm_rotated_matrix"
MTP_MTP_NORM_ROTATED_MATRIX_KEY = "mtp.mtp_norm_rotated_matrix"


# ---------------------------------------------------------------------------
# safetensors index helpers
# ---------------------------------------------------------------------------

def _load_index(model_dir: str) -> Tuple[Dict[str, str], Dict]:
    """Return (weight_map, full_index_dict) from `model.safetensors.index.json`.
    If the model has no index (single-file safetensors), build an in-memory map.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("weight_map", {})), data
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="pt", device="cpu") as f:
            wm = {k: "model.safetensors" for k in f.keys()}
        return wm, {"weight_map": wm, "metadata": {}}
    raise FileNotFoundError(f"No safetensors found under {model_dir}")


def _open_tensor(model_dir: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    """Load a single tensor lazily from the appropriate shard."""
    if key not in weight_map:
        raise KeyError(f"Tensor `{key}` not in weight_map of {model_dir}")
    shard = os.path.join(model_dir, weight_map[key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# ---------------------------------------------------------------------------
# math primitives — mirror rotation.py conventions
#   • input-side rotation:  W = W @ Q
#   • output-side rotation: W = Q.T @ W
#   • Qwen3.5 RMSNorm uses (1 + γ); fuse multiplies weights by (1 + γ) along ic
# ---------------------------------------------------------------------------

def _qwen35_norm_scale(norm_weight: torch.Tensor) -> torch.Tensor:
    """Qwen3.5 RMSNorm uses `(1 + weight) · x` scaling."""
    return (1.0 + norm_weight.double())


def _rotated_norm_matrix(scale: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Return T where `(x @ Q) @ T == (x * scale) @ Q` for row-vector hidden states."""
    rotated = torch.matmul(Q.T, scale.to(Q.dtype).unsqueeze(1) * Q)
    return rotated.to(torch.float32).contiguous()


def _fuse_norm_into_weight_in(weight: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    """Apply `W <- W * scale` along the input dim (last dim of weight, oc x ic)."""
    dtype = weight.dtype
    W = weight.double()
    scale = _qwen35_norm_scale(norm_w).to(W.dtype)
    return (W * scale.unsqueeze(0)).to(dtype)


def _rotate_in(weight: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Input-side rotation: W = W @ Q. Q is float64 on CPU."""
    dtype = weight.dtype
    W = weight.double()
    return torch.matmul(W, Q).to(dtype)


def _rotate_out(weight: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Output-side rotation: W = Q.T @ W."""
    dtype = weight.dtype
    W = weight.double()
    return torch.matmul(Q.T, W).to(dtype)


# ---------------------------------------------------------------------------
# MTP rotate driver
# ---------------------------------------------------------------------------

def rotate_qwen3_5_mtp(
    src_model_dir: str,
    dst_model_dir: str,
    Q_block: torch.Tensor,
    *,
    is_moe: bool,
    has_linear_attn: bool = False,
    lm_head_key: str = "lm_head.weight",
    pre_lm_head_norm_key: str = "model.norm.weight",
) -> Optional[str]:
    """
    Read mtp.* + un-fused lm_head + un-fused model.norm from `src_model_dir`,
    apply Qwen3.5 fuse + rotate(Q_block), write a sidecar `mtp_rotated.safetensors`
    into `dst_model_dir`, and patch the index.json.

    Returns the absolute path of the sidecar file (or None if no MTP found).
    """
    Q = Q_block.detach().to(dtype=torch.float64, device="cpu").contiguous()

    src_wm, _ = _load_index(src_model_dir)
    mtp_keys = [k for k in src_wm if k.startswith("mtp.")]
    if not mtp_keys:
        log.info("MTP rotate: no `mtp.*` tensors found in %s — skipping.", src_model_dir)
        return None

    log.info("MTP rotate: %d mtp tensors found; is_moe=%s", len(mtp_keys), is_moe)

    out_state: Dict[str, torch.Tensor] = {}
    H = int(Q.shape[0])

    # --- 0. Preserve source-specific final-norm scales and build the
    #        dedicated MTP head BEFORE we mutate anything ---
    #     mtp.lm_head_weight = (lm_head_orig * (1 + mtp.norm_orig)) @ Q
    if lm_head_key not in src_wm:
        # tied-embedding path; fall back to embed_tokens.weight (rare for Qwen3.5)
        candidates = [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ]
        lm_head_key = next((c for c in candidates if c in src_wm), lm_head_key)
        log.warn("MTP rotate: lm_head not found, fell back to %s", lm_head_key)
    if pre_lm_head_norm_key not in src_wm:
        candidates = [
            "model.language_model.norm.weight",
            "language_model.norm.weight",
            "model.norm.weight",
        ]
        pre_lm_head_norm_key = next((c for c in candidates if c in src_wm), pre_lm_head_norm_key)
        log.warn("MTP rotate: pre-lm-head norm not found, fell back to %s", pre_lm_head_norm_key)
    lm_head_orig = _open_tensor(src_model_dir, src_wm, lm_head_key)
    lm_norm_orig = _open_tensor(src_model_dir, src_wm, pre_lm_head_norm_key)
    mtp_norm_orig = _open_tensor(src_model_dir, src_wm, "mtp.norm.weight")
    lm_norm_scale = _qwen35_norm_scale(lm_norm_orig)
    mtp_norm_scale = _qwen35_norm_scale(mtp_norm_orig)
    out_state[MTP_LANGUAGE_MODEL_NORM_SCALE_KEY] = lm_norm_scale.to(lm_norm_orig.dtype).contiguous()
    out_state[MTP_MTP_NORM_SCALE_KEY] = mtp_norm_scale.to(mtp_norm_orig.dtype).contiguous()
    out_state[MTP_LANGUAGE_MODEL_NORM_ROTATED_MATRIX_KEY] = _rotated_norm_matrix(lm_norm_scale, Q)
    out_state[MTP_MTP_NORM_ROTATED_MATRIX_KEY] = _rotated_norm_matrix(mtp_norm_scale, Q)
    head_dtype = lm_head_orig.dtype
    head_w = _fuse_norm_into_weight_in(lm_head_orig, mtp_norm_orig)
    head_w = _rotate_in(head_w, Q)
    out_state["mtp.lm_head_weight"] = head_w.to(head_dtype).contiguous()

    # --- 1. fc + pre_fc_norms ---
    fc_w_orig = _open_tensor(src_model_dir, src_wm, "mtp.fc.weight")  # (H, 2H)
    if fc_w_orig.shape[-1] != 2 * H:
        raise ValueError(
            f"MTP fc.weight last-dim {fc_w_orig.shape[-1]} != 2*H ({2*H}). "
            "Expected concat([norm_e(e); norm_h(h)])."
        )
    pre_h = _open_tensor(src_model_dir, src_wm, "mtp.pre_fc_norm_hidden.weight")
    pre_e = _open_tensor(src_model_dir, src_wm, "mtp.pre_fc_norm_embedding.weight")
    fc_dtype = fc_w_orig.dtype
    fc64 = fc_w_orig.double()
    sh = _qwen35_norm_scale(pre_h)  # (H,)
    se = _qwen35_norm_scale(pre_e)  # (H,)
    # Qwen3.5 MTP concatenates [embedding, hidden] before fc.
    fc64[:, :H] = fc64[:, :H] * se.unsqueeze(0)
    fc64[:, H:] = fc64[:, H:] * sh.unsqueeze(0)
    # input-side rotate: each half is in the LLM rotated space
    fc64[:, :H] = torch.matmul(fc64[:, :H], Q)
    fc64[:, H:] = torch.matmul(fc64[:, H:], Q)
    # output-side rotate: fc output feeds mtp.layers, also rotated
    fc64 = torch.matmul(Q.T, fc64)
    out_state["mtp.fc.weight"] = fc64.to(fc_dtype).contiguous()

    # zero the norms (1 + 0) = identity post-fuse
    out_state["mtp.pre_fc_norm_hidden.weight"] = torch.zeros_like(pre_h)
    out_state["mtp.pre_fc_norm_embedding.weight"] = torch.zeros_like(pre_e)
    out_state["mtp.norm.weight"] = torch.zeros_like(mtp_norm_orig)

    # --- 2. mtp.layers.* ---
    layer_indices = sorted({
        int(k.split(".")[2]) for k in mtp_keys if k.startswith("mtp.layers.")
    })
    log.info("MTP rotate: %d layer block(s) to process", len(layer_indices))

    for i in layer_indices:
        prefix = f"mtp.layers.{i}"

        # 2a. fuse input_layernorm into self_attn / linear_attn input projs;
        #     fuse post_attention_layernorm into mlp inputs.
        in_ln = _open_tensor(src_model_dir, src_wm, f"{prefix}.input_layernorm.weight")
        post_ln = _open_tensor(src_model_dir, src_wm, f"{prefix}.post_attention_layernorm.weight")

        attn_input_keys: List[str] = []
        for sub in ("q_proj", "k_proj", "v_proj"):
            k = f"{prefix}.self_attn.{sub}.weight"
            if k in src_wm:
                attn_input_keys.append(k)
        if has_linear_attn:
            for sub in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                k = f"{prefix}.linear_attn.{sub}.weight"
                if k in src_wm:
                    attn_input_keys.append(k)

        for k in attn_input_keys:
            w = _open_tensor(src_model_dir, src_wm, k)
            w = _fuse_norm_into_weight_in(w, in_ln)
            out_state[k] = _rotate_in(w, Q).contiguous()

        # 2b. attention output projections: output-side rotation, no fuse
        for sub in ("o_proj",):
            k = f"{prefix}.self_attn.{sub}.weight"
            if k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, k)
                out_state[k] = _rotate_out(w, Q).contiguous()
        if has_linear_attn:
            k = f"{prefix}.linear_attn.out_proj.weight"
            if k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, k)
                out_state[k] = _rotate_out(w, Q).contiguous()

        # 2c. mlp / experts
        if is_moe:
            # router gate: fuse post_ln (a learned multiplier on hidden), then input rotate
            gate_k = f"{prefix}.mlp.gate.weight"
            if gate_k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, gate_k)
                w = _fuse_norm_into_weight_in(w, post_ln)
                out_state[gate_k] = _rotate_in(w, Q).contiguous()

            # shared expert (dense MLP)
            for sub in ("gate_proj", "up_proj"):
                k = f"{prefix}.mlp.shared_expert.{sub}.weight"
                if k in src_wm:
                    w = _open_tensor(src_model_dir, src_wm, k)
                    w = _fuse_norm_into_weight_in(w, post_ln)
                    out_state[k] = _rotate_in(w, Q).contiguous()
            k = f"{prefix}.mlp.shared_expert.down_proj.weight"
            if k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, k)
                out_state[k] = _rotate_out(w, Q).contiguous()
            seg_k = f"{prefix}.mlp.shared_expert_gate.weight"
            if seg_k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, seg_k)
                w = _fuse_norm_into_weight_in(w, post_ln)
                out_state[seg_k] = _rotate_in(w, Q).contiguous()

            # routed experts
            expert_indices = sorted({
                int(k.split(".")[5])
                for k in mtp_keys
                if k.startswith(f"{prefix}.mlp.experts.")
            })
            log.info("MTP rotate: layer %d has %d routed experts", i, len(expert_indices))
            for ei in expert_indices:
                ep = f"{prefix}.mlp.experts.{ei}"
                for sub in ("gate_proj", "up_proj"):
                    k = f"{ep}.{sub}.weight"
                    if k in src_wm:
                        w = _open_tensor(src_model_dir, src_wm, k)
                        w = _fuse_norm_into_weight_in(w, post_ln)
                        out_state[k] = _rotate_in(w, Q).contiguous()
                k = f"{ep}.down_proj.weight"
                if k in src_wm:
                    w = _open_tensor(src_model_dir, src_wm, k)
                    out_state[k] = _rotate_out(w, Q).contiguous()
        else:
            for sub in ("gate_proj", "up_proj"):
                k = f"{prefix}.mlp.{sub}.weight"
                if k in src_wm:
                    w = _open_tensor(src_model_dir, src_wm, k)
                    w = _fuse_norm_into_weight_in(w, post_ln)
                    out_state[k] = _rotate_in(w, Q).contiguous()
            k = f"{prefix}.mlp.down_proj.weight"
            if k in src_wm:
                w = _open_tensor(src_model_dir, src_wm, k)
                out_state[k] = _rotate_out(w, Q).contiguous()

        # 2d. zero fused norms; preserve q_norm/k_norm (they are head-dim norms,
        #     not part of the residual stream, so they aren't fused/rotated).
        out_state[f"{prefix}.input_layernorm.weight"] = torch.zeros_like(in_ln)
        out_state[f"{prefix}.post_attention_layernorm.weight"] = torch.zeros_like(post_ln)
        for sub in ("q_norm", "k_norm"):
            k = f"{prefix}.self_attn.{sub}.weight"
            if k in src_wm:
                out_state[k] = _open_tensor(src_model_dir, src_wm, k).clone()

    # --- 3. write sidecar + patch index ---
    os.makedirs(dst_model_dir, exist_ok=True)
    sidecar_path = os.path.join(dst_model_dir, MTP_SIDECAR_FILENAME)
    save_file(out_state, sidecar_path, metadata={"format": "pt", "produced_by": "gptqmodel.rotation.mtp"})
    log.info("MTP rotate: wrote %d tensors to %s", len(out_state), sidecar_path)

    _patch_index_for_sidecar(dst_model_dir, out_state.keys(), MTP_SIDECAR_FILENAME, sidecar_path)
    return sidecar_path


def _patch_index_for_sidecar(
    dst_model_dir: str,
    new_keys: Iterable[str],
    sidecar_filename: str,
    sidecar_abs_path: str,
) -> None:
    """Update / create model.safetensors.index.json so loaders find the new tensors."""
    index_path = os.path.join(dst_model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        # Single-shard layout: convert to indexed for consistency.
        single = os.path.join(dst_model_dir, "model.safetensors")
        index_data = {"metadata": {}, "weight_map": {}}
        if os.path.exists(single):
            with safe_open(single, framework="pt", device="cpu") as f:
                for k in f.keys():
                    index_data["weight_map"][k] = "model.safetensors"

    wm = index_data.setdefault("weight_map", {})
    for k in new_keys:
        # If the original (un-rotated) key existed in an earlier shard (e.g.
        # because save_pretrained auto-included mtp.*), redirect it to our sidecar
        # so the rotated version wins.
        wm[k] = sidecar_filename

    meta = index_data.setdefault("metadata", {})
    try:
        meta["total_size"] = int(meta.get("total_size", 0)) + os.path.getsize(sidecar_abs_path)
    except OSError:
        pass

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, sort_keys=True)
        f.write("\n")
    log.info("MTP rotate: patched %s with %d new keys -> %s",
             os.path.basename(index_path), len(list(new_keys)), sidecar_filename)
