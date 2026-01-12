# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
import typing

import torch
from transformers import PreTrainedModel

from ...utils.logger import setup_logger
from ...utils.model import get_module_by_name_prefix
from ...utils.torch import torch_empty_cache
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix


log = setup_logger()


def fuse_ln_linear(
        layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def reset_ln(ln):
    W_norm = ln.weight.data
    ln.weight.data = torch.ones_like(W_norm)
    if hasattr(ln, "bias"):
        ln.bias.data.zero_()


def fuse_merger_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        w_o, w_i = W_.shape
        size = layernorm.weight.shape[0]
        linear.weight.data = (
            (W_.view(w_o, -1, size) * layernorm.weight.double())
            .to(linear_dtype)
            .view(w_o, w_i)
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features,
                                dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_.view(w_o, -1, size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def fuse_layer_norms(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    layers , _ = get_module_by_name_prefix(model, layers_node)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def fuse_layer_norms_moe(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    layers , _ = get_module_by_name_prefix(model, layers_node)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.gate]
            )
        for i in range(128):
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.experts[i].gate_proj, layer.mlp.experts[i].up_proj]
            )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def bake_mean_into_linear_conv(linear: torch.nn.Linear, embed_dims) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    weight_shape = linear.weight.data.shape
    W_ = linear.weight.data.double().view(embed_dims, -1)
    linear.weight.data = W_ - W_.mean(dim=0, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype).view(weight_shape)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_vision_layer_norms_qwen2_5_vl(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    model = model.eval()
    for layer in model.visual.blocks:
        fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        reset_ln(layer.norm1)
        fuse_ln_linear(
            layer.norm2, [layer.mlp.gate_proj, layer.mlp.up_proj])
        reset_ln(layer.norm2)
    fuse_merger_linear(model.visual.merger.ln_q, [
                    model.visual.merger.mlp[0]])
    reset_ln(model.visual.merger.ln_q)
    return model


def fuse_vision_layer_norms_qwen3_vl(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    model = model.eval()
    bake_mean_into_linear_conv(model.visual.patch_embed.proj, model.visual.blocks[0].attn.qkv.in_features)
    for W in [model.visual.pos_embed]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)
                        ).to(W.weight.data.dtype)
    for layer in model.visual.blocks:
        fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        reset_ln(layer.norm1)
        bake_mean_into_linear(layer.attn.proj)
        fuse_ln_linear(layer.norm2, [layer.mlp.linear_fc1])
        reset_ln(layer.norm2)
        bake_mean_into_linear(layer.mlp.linear_fc2)
    fuse_merger_linear(model.visual.merger.norm, [
                    model.visual.merger.linear_fc1])
    reset_ln(model.visual.merger.norm)
    return model


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_conv(layer, Q_v, embed_dims, device):
    dtype = layer.weight.dtype
    weight_shape = layer.weight.data.shape
    layer = layer.to(device)
    layer.weight.data = (
        torch.matmul(Q_v.T, layer.weight.data.double().view(
            embed_dims, -1)).to(dtype=dtype, device=device).view(weight_shape)
    )
    if layer.bias is not None:
        layer.bias.data = torch.matmul(layer.bias.data.double(), Q_v).to(dtype=dtype, device=device)


def rotate_visual_merger(model, Q: torch.Tensor, device) -> None:
    # Rotate the head.
    dtype = model.visual.merger.mlp[0].weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.mlp[0].weight.shape

    W_ = model.visual.merger.mlp[0].weight.to(
        dtype=torch.float64, device=device).reshape(o_shape, -1, q_shape)
    model.visual.merger.mlp[0].weight.data = torch.matmul(W_, Q).to(
        dtype=dtype, device=device).reshape(o_shape, i_shape).contiguous()


def rotate_visual_merger_qwen3_vl(model, Q: torch.Tensor, device) -> None:
    # Rotate the head.
    dtype = model.visual.merger.linear_fc1.weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.linear_fc1.weight.shape

    W_ = model.visual.merger.linear_fc1.weight.to(
        dtype=torch.float64, device=device).reshape(o_shape, -1, q_shape)
    model.visual.merger.linear_fc1.weight.data = torch.matmul(W_, Q).to(
        dtype=dtype, device=device).reshape(o_shape, i_shape).contiguous()


def rotate_qwen3_vl_embeddings(model, Q, device) -> None:
    for W in [model.language_model.embed_tokens]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    W_ = model.visual.merger.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
    model.visual.merger.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
    if model.visual.merger.linear_fc2.bias is not None:
        b = model.visual.merger.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
        model.visual.merger.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")

    for layer in model.visual.deepstack_merger_list:
        W_ = layer.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
        layer.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
        if layer.linear_fc2.bias is not None:
            b = layer.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
            layer.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")


def rotate_qwen2_5_vl_embeddings(model, Q, device) -> None:
    for W in [model.language_model.embed_tokens]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    W_ = model.visual.merger.mlp[2].weight.data.to(dtype=torch.float64, device=device)
    model.visual.merger.mlp[2].weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
    if model.visual.merger.mlp[2].bias is not None:
        b = model.visual.merger.mlp[2].bias.data.to(dtype=torch.float64, device=device)
        model.visual.merger.mlp[2].bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")


def rotate_qwen3_vl_vision_embeddings(model, Q, device) -> None:
    for W in [model.visual.pos_embed]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def get_embeddings(model) -> list[torch.nn.Module]:
    return [model.model.embed_tokens]


def rotate_embeddings(model, Q, device) -> None:
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, device) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    try:
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    except:
        for W in [layer.attn.qkv]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, device) -> None:
    # Rotate output matrix of the self-attention layer.
    try:
        W = layer.self_attn.o_proj
    except:
        W = layer.attn.proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_input_qwen3_vl(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.linear_fc1]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_input_moe(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.gate]
    for i in range(128):
        mlp_inputs.append(layer.mlp.experts[i].gate_proj)
        mlp_inputs.append(layer.mlp.experts[i].up_proj)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_qwen3_vl(layer, Q, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.linear_fc2
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_moe(layer, Q, device):
    # Rotate the MLP output weights and bias.
    for i in range(128):
        W = layer.mlp.experts[i].down_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q, device, lm_head_name) -> None:
    # Rotate the head.
    W, _ = get_module_by_name_prefix(model, lm_head_name)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


@torch.inference_mode()
def rotate_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                 layers_node: str, Q=None):
    Q = (
        get_orthogonal_matrix(
            model.config.hidden_size, rotate_mode, device
        )
        if Q is None
        else Q
    )
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
        # rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q


@torch.inference_mode()
def rotate_model_qwen2_5_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    Q = (
        get_orthogonal_matrix(
            model.config.text_config.hidden_size, rotate_mode, device
        )
        if Q is None
        else Q
    )
    config = model.config
    num_heads = config.text_config.num_attention_heads
    model_dim = config.text_config.hidden_size
    head_dim = model_dim // num_heads

    rotate_qwen2_5_vl_embeddings(model, Q, device)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
    return model, Q


@torch.inference_mode()
def rotate_model_qwen3_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    Q = (
        get_orthogonal_matrix(
            model.config.text_config.hidden_size, rotate_mode, device
        )
        if Q is None
        else Q
    )
    config = model.config
    num_heads = config.text_config.num_attention_heads
    model_dim = config.text_config.hidden_size
    head_dim = model_dim // num_heads

    rotate_qwen3_vl_embeddings(model, Q, device)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
    return model, Q


@torch.inference_mode()
def rotate_model_moe(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    Q = (
        get_orthogonal_matrix(
            model.config.hidden_size, rotate_mode, device
        )
        if Q is None
        else Q
    )

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input_moe(layer, Q, device)
        rotate_mlp_output_moe(layer, Q, device)
        # rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q


@torch.inference_mode()
def rotate_vision_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                        layers_node: str, Q=None):
    num_heads = model.visual.blocks[0].attn.num_heads
    Q = (
        get_orthogonal_matrix(
            model.visual.blocks[0].attn.qkv.in_features, rotate_mode, device
        )
        if Q is None
        else Q
    )
    rotate_conv(
            model.visual.patch_embed.proj,
            Q,
            model.visual.blocks[0].attn.qkv.in_features,
            device,
    )
    torch_empty_cache()
    for idx, layer in enumerate(log.pb(model.visual.blocks).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
    rotate_visual_merger(model, Q, device)
    return model, Q


@torch.inference_mode()
def rotate_vision_model_qwen3_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                        layers_node: str, Q=None):
    num_heads = model.visual.blocks[0].attn.num_heads
    Q = (
        get_orthogonal_matrix(
            model.visual.blocks[0].attn.qkv.in_features, rotate_mode, device
        )
        if Q is None
        else Q
    )
    rotate_conv(
            model.visual.patch_embed.proj,
            Q,
            model.visual.blocks[0].attn.qkv.in_features,
            device,
    )
    rotate_qwen3_vl_vision_embeddings(model, 
                                      Q, 
                                      device) 
    torch_empty_cache()
    for idx, layer in enumerate(log.pb(model.visual.blocks).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input_qwen3_vl(layer, Q, device)
        rotate_mlp_output_qwen3_vl(layer, Q, device)
    rotate_visual_merger_qwen3_vl(model, Q, device)
    return model, Q