# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ...nn_modules.hooked_linear import StopForward
from ...utils.device import get_device
from ...utils.model import move_to


class Gemma4NaiveExpert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, act_fn, gate_weight=None, up_weight=None, down_weight=None):
        super().__init__()

        self.act_fn = act_fn
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        if gate_weight is not None:
            self.gate_proj.weight = nn.Parameter(gate_weight)
        if up_weight is not None:
            self.up_proj.weight = nn.Parameter(up_weight)
        if down_weight is not None:
            self.down_proj.weight = nn.Parameter(down_weight)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma4NaiveExperts(nn.Module):
    """
    Split Gemma4TextExperts packed 3-D Parameter tensors (gate_up_proj, down_proj)
    into per-expert nn.Linear modules so that GPTQ can quantize them individually.

    Original Gemma4TextExperts stores:
        gate_up_proj: Parameter[num_experts, 2*moe_intermediate_size, hidden_size]
        down_proj:    Parameter[num_experts, hidden_size, moe_intermediate_size]

    After conversion the module exposes:
        experts[i].gate_proj: Linear(hidden_size -> moe_intermediate_size)
        experts[i].up_proj:   Linear(hidden_size -> moe_intermediate_size)
        experts[i].down_proj: Linear(moe_intermediate_size -> hidden_size)
    """

    def __init__(self, config=None, ori_experts=None):
        super().__init__()

        if ori_experts is not None:
            self.num_experts = ori_experts.num_experts
            self.hidden_dim = ori_experts.hidden_dim
            self.intermediate_dim = ori_experts.intermediate_dim
            self.act_fn = ori_experts.act_fn
        else:
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = ACT2FN[config.hidden_activation]

        self.experts = nn.ModuleList()

        with torch.no_grad():
            for i in range(self.num_experts):
                if ori_experts is not None:
                    gate_up = ori_experts.gate_up_proj[i]
                    mid = gate_up.shape[0] // 2
                    gate_weight = gate_up[:mid, :].contiguous()
                    up_weight = gate_up[mid:, :].contiguous()
                    down_weight = ori_experts.down_proj[i].contiguous()
                    dtype = gate_weight.dtype
                    device = gate_weight.device
                else:
                    dtype = torch.float32
                    device = "cpu"
                    gate_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    up_weight = torch.empty(self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
                    down_weight = torch.empty(self.hidden_dim, self.intermediate_dim, dtype=dtype, device=device)

                self.experts.append(
                    Gemma4NaiveExpert(
                        hidden_dim=self.hidden_dim,
                        intermediate_dim=self.intermediate_dim,
                        act_fn=self.act_fn,
                        gate_weight=gate_weight.to(dtype=dtype, device=device),
                        up_weight=up_weight.to(dtype=dtype, device=device),
                        down_weight=down_weight.to(dtype=dtype, device=device),
                    )
                )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Gemma4MoELifecycleHooks(GateUpDownMoELifecycleHooks):
    def get_moe_block(self, layer_module: nn.Module, model_class: type):
        if (
            getattr(layer_module, "enable_moe_block", False)
            and hasattr(layer_module, "experts")
            and hasattr(layer_module, "router")
            and hasattr(layer_module, "pre_feedforward_layernorm_2")
        ):
            return layer_module
        return None

    def get_experts_module(self, moe_block: nn.Module, model_class: type):
        experts_container = getattr(moe_block, "experts", None)
        if experts_container is None:
            return None
        return getattr(experts_container, "experts", None)

    def forward_to_all_experts(
        self,
        moe_block: nn.Module,
        hidden_states: torch.Tensor,
        processor,
        subset,
        original_forward: callable,
        model_class: type,
        module_looper,
        moe_block_prefix=None,
        replica_module=None,
        **kwargs,
    ):
        if not processor or not original_forward:
            raise ValueError("Missing processor or original_forward")

        experts_module = self.get_experts_module(moe_block, model_class)
        if experts_module is None or moe_block_prefix is None:
            return original_forward(hidden_states, **kwargs)

        def get_callable_module(key: str):
            if replica_module is not None:
                current = replica_module
                for part in key.split("."):
                    if hasattr(current, part):
                        current = getattr(current, part)
                    elif hasattr(current, "__getitem__") and part.isdigit():
                        current = current[int(part)]
                    else:
                        current = None
                        break
                if current is not None:
                    return current
            return subset.get(key)

        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            hidden_states_flat = hidden_states

        pre_ff_ln_2 = moe_block.pre_feedforward_layernorm_2
        expert_input_base = pre_ff_ln_2(hidden_states_flat)

        expert_count = 0
        stop_forward_raised = False
        experts_attr_name = self.get_experts_module_name(moe_block)

        for expert_idx, expert in enumerate(experts_module):
            gate_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.gate_proj_name}"
            up_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.up_proj_name}"
            down_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.down_proj_name}"

            if gate_key not in subset and up_key not in subset and down_key not in subset:
                continue

            gate_module_ref = getattr(expert, self.gate_proj_name, None)
            expert_device = get_device(gate_module_ref if gate_module_ref is not None else expert)
            expert_input = move_to(expert_input_base, expert_device)

            try:
                if down_key in subset:
                    gate_module = getattr(expert, self.gate_proj_name)
                    up_module = getattr(expert, self.up_proj_name)
                    gate_out = gate_module(expert_input)
                    up_out = up_module(expert_input)
                    intermediate = expert.act_fn(gate_out) * up_out
                    get_callable_module(down_key)(intermediate)
                    del gate_out, up_out, intermediate
                    expert_count += 1
                else:
                    called_any = False
                    if gate_key in subset:
                        get_callable_module(gate_key)(expert_input)
                        called_any = True
                    if up_key in subset:
                        get_callable_module(up_key)(expert_input)
                        called_any = True
                    if called_any:
                        expert_count += 1
            except StopForward:
                stop_forward_raised = True
            finally:
                del expert_input

        if stop_forward_raised:
            raise StopForward()

        if expert_count > 0:
            module_looper._set_processor_hooks_paused(processor, True)
            try:
                result = original_forward(hidden_states, **kwargs)
            finally:
                module_looper._set_processor_hooks_paused(processor, False)
        else:
            result = original_forward(hidden_states, **kwargs)

        return result


def _install_rotary_fix_hooks(model, config):
    text_config = getattr(config, "text_config", config)

    layer_types = getattr(text_config, "layer_types", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if not layer_types or hidden_size is None or len(set(layer_types)) <= 1:
        return

    text_model = model
    for attr in ("model", "language_model"):
        child = getattr(text_model, attr, None)
        if child is not None:
            text_model = child
    if not hasattr(text_model, "rotary_emb") or not hasattr(text_model, "layers"):
        return

    rotary_emb = text_model.rotary_emb

    expected_dims: dict = {}
    dummy_hidden = torch.zeros(1, 1, hidden_size)
    dummy_pos = torch.zeros(1, 1, dtype=torch.long)
    for layer_type in set(layer_types):
        with torch.no_grad():
            cos, _ = rotary_emb(dummy_hidden, dummy_pos, layer_type)
        expected_dims[layer_type] = cos.shape[-1]

    def _make_hook(layer_type, expected_dim):
        def hook(module, args, kwargs):
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                return args, kwargs
            cos, _ = position_embeddings
            if cos.shape[-1] == expected_dim:
                return args, kwargs
            position_ids = kwargs.get("position_ids")
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if position_ids is not None and hidden_states is not None:
                with torch.no_grad():
                    new_position_embeddings = rotary_emb(hidden_states, position_ids, layer_type)
                kwargs = dict(kwargs)
                kwargs["position_embeddings"] = new_position_embeddings
            return args, kwargs

        return hook

    for i, layer in enumerate(text_model.layers):
        layer_type = layer_types[i]
        layer.register_forward_pre_hook(
            _make_hook(layer_type, expected_dims[layer_type]),
            with_kwargs=True,
        )


def convert_gemma4_experts(module, config):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    to_replace = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, Gemma4TextExperts):
            to_replace.append((name, sub_module))

    for name, sub_module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        new_module = Gemma4NaiveExperts(config=config, ori_experts=sub_module)
        setattr(parent, child_name, new_module)

    torch.cuda.empty_cache()
    return module


class Gemma4TextQModel(BaseQModel):
    # Gemma4 is a dual-architecture model: dense (num_experts=None) or MoE (num_experts>0).
    # dynamic_expert_index triggers MoE expansion when num_experts > 0; dense config returns
    # None/0 from get_num_experts, causing build_moe_modules_if_need to use the dense path.
    dynamic_expert_index = "num_experts"
    support_offload_to_disk = False
    pre_lm_head_norm_module = "model.norm"
    layer_modules_strict = False
    moe_lifecycle_hooks = Gemma4MoELifecycleHooks()

    @classmethod
    def get_num_experts(cls, model_config):
        # For dense Gemma4 config, num_experts is None; return 0 to signal no MoE expansion.
        num_experts = super().get_num_experts(model_config)
        return num_experts if num_experts else 0

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
            "post_feedforward_layernorm_1": ("post_feedforward_layernorm_1:!",),
            "pre_feedforward_layernorm_2": ("pre_feedforward_layernorm_2:!",),
            "post_feedforward_layernorm_2": ("post_feedforward_layernorm_2:!",),
            "router": ("router:!",),
            "experts:moe": {
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

            gemma4_modeling.Gemma4TextExperts = Gemma4NaiveExperts

    def after_model_load(self, model, load_quantized_model):
        self.model = super().after_model_load(model, load_quantized_model)

        if not load_quantized_model:
            self.model = convert_gemma4_experts(self.model, self.config)
            _install_rotary_fix_hooks(self.model, self.config)

        return self.model



class Gemma4ForConditionalGenerationGPTQ(BaseQModel):
    support_batch_quantize = False
    # Gemma4 is a dual-architecture model: dense (num_experts=None) or MoE (num_experts>0).
    # dynamic_expert_index triggers MoE expansion when num_experts > 0; dense config returns
    # None/0 from get_num_experts, causing build_moe_modules_if_need to use the dense path.
    dynamic_expert_index = "num_experts"
    support_offload_to_disk = False
    pre_lm_head_norm_module = "model.language_model.norm"
    layer_modules_strict = False
    moe_lifecycle_hooks = Gemma4MoELifecycleHooks()

    @classmethod
    def get_num_experts(cls, model_config):
        # For dense Gemma4 config, num_experts is None; return 0 to signal no MoE expansion.
        num_experts = super().get_num_experts(model_config)
        return num_experts if num_experts else 0

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
            "post_feedforward_layernorm_1": ("post_feedforward_layernorm_1:!",),
            "pre_feedforward_layernorm_2": ("pre_feedforward_layernorm_2:!",),
            "post_feedforward_layernorm_2": ("post_feedforward_layernorm_2:!",),
            "router": ("router:!",),
            "experts:moe": {
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    lm_head_module = "model.lm_head"

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

            gemma4_modeling.Gemma4TextExperts = Gemma4NaiveExperts

    def after_model_load(self, model, load_quantized_model):
        self.model = super().after_model_load(model, load_quantized_model)

        if not load_quantized_model:
            self.model = convert_gemma4_experts(self.model, self.config)
            _install_rotary_fix_hooks(self.model, self.config)

        return self.model