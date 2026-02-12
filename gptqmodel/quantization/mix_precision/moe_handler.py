# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class MoEHandler:
    """MoE结构识别与统一处理
    
    负责识别MoE结构，确保MoE组内所有专家使用统一的量化bit
    """
    
    MOE_EXPERT_PATTERN = r"\.experts\.\d+\."
    MOE_SHARED_EXPERT_PATTERN = r"\.shared_experts\."
    
    MOE_CONTAINER_TYPES = [
        "MoE", "Experts", "ExpertBlock", "MixtralSparseMoeBlock",
        "Qwen2MoeSparseMoeBlock", "DeepseekV2MoE", "Glm4MoeTopKRouter",
    ]
    
    @classmethod
    def is_moe_module(cls, module_name: str) -> bool:
        """判断是否为MoE模块
        
        Args:
            module_name: 模块完整名称
            
        Returns:
            是否为MoE模块
        """
        if not module_name:
            return False
        
        if re.search(cls.MOE_EXPERT_PATTERN, module_name):
            return True
        
        if re.search(cls.MOE_SHARED_EXPERT_PATTERN, module_name):
            return True
        
        return False
    
    @classmethod
    def is_routed_expert(cls, module_name: str) -> bool:
        """判断是否为路由专家（非共享专家）
        
        Args:
            module_name: 模块完整名称
            
        Returns:
            是否为路由专家
        """
        if not module_name:
            return False
        return bool(re.search(cls.MOE_EXPERT_PATTERN, module_name))
    
    @classmethod
    def is_shared_expert(cls, module_name: str) -> bool:
        """判断是否为共享专家
        
        Args:
            module_name: 模块完整名称
            
        Returns:
            是否为共享专家
        """
        if not module_name:
            return False
        return bool(re.search(cls.MOE_SHARED_EXPERT_PATTERN, module_name))
    
    @classmethod
    def get_moe_group_key(cls, module_name: str) -> Optional[str]:
        """提取MoE组标识
        
        将同一MoE层内的所有专家归为同一组
        
        例如:
            "model.layers.1.mlp.experts.0.gate_proj" 
            -> "model.layers.1.mlp.experts"
            
            "model.layers.1.mlp.shared_experts.gate_proj"
            -> "model.layers.1.mlp.shared_experts"
        
        Args:
            module_name: 模块完整名称
            
        Returns:
            MoE组标识，如果不是MoE模块则返回None
        """
        if not module_name:
            return None
        
        if ".experts." in module_name:
            idx = module_name.find(".experts.")
            return module_name[:idx + len(".experts")]
        
        if ".shared_experts." in module_name:
            idx = module_name.find(".shared_experts.")
            return module_name[:idx + len(".shared_experts")]
        
        return None
    
    @classmethod
    def get_moe_layer_key(cls, module_name: str) -> Optional[str]:
        """提取MoE层标识（包含路由专家和共享专家）
        
        将同一MoE层内的路由专家和共享专家归为同一层
        
        例如:
            "model.layers.1.mlp.experts.0.gate_proj" 
            -> "model.layers.1.mlp"
            
            "model.layers.1.mlp.shared_experts.gate_proj"
            -> "model.layers.1.mlp"
        
        Args:
            module_name: 模块完整名称
            
        Returns:
            MoE层标识，如果不是MoE模块则返回None
        """
        if not module_name:
            return None
        
        if ".experts." in module_name:
            idx = module_name.find(".experts.")
            return module_name[:idx]
        
        if ".shared_experts." in module_name:
            idx = module_name.find(".shared_experts.")
            return module_name[:idx]
        
        return None
    
    @classmethod
    def collect_moe_modules(cls, model: nn.Module) -> Dict[str, List[str]]:
        """收集所有MoE组及其成员模块
        
        Args:
            model: 模型实例
            
        Returns:
            {moe_group_key: [module_name1, module_name2, ...]}
        """
        moe_groups = defaultdict(list)
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                group_key = cls.get_moe_group_key(name)
                if group_key:
                    moe_groups[group_key].append(name)
        
        return dict(moe_groups)
    
    @classmethod
    def collect_moe_layers(cls, model: nn.Module) -> Dict[str, Dict[str, List[str]]]:
        """收集所有MoE层及其内部结构
        
        按MoE层组织，区分路由专家和共享专家
        
        Args:
            model: 模型实例
            
        Returns:
            {moe_layer_key: {"experts": [...], "shared_experts": [...]}}
        """
        moe_layers = defaultdict(lambda: {"experts": [], "shared_experts": []})
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                layer_key = cls.get_moe_layer_key(name)
                if layer_key:
                    if cls.is_routed_expert(name):
                        moe_layers[layer_key]["experts"].append(name)
                    elif cls.is_shared_expert(name):
                        moe_layers[layer_key]["shared_experts"].append(name)
        
        return dict(moe_layers)
    
    @classmethod
    def unify_moe_bits(
        cls,
        bit_allocation: Dict[str, int],
        moe_groups: Dict[str, List[str]],
        strategy: str = "max",
    ) -> Dict[str, int]:
        """统一MoE组内所有专家的bit
        
        Args:
            bit_allocation: 原始bit分配 {module_name: bits}
            moe_groups: MoE组映射 {moe_group_key: [module_names]}
            strategy: 统一策略
                - "max": 使用组内最高bit（最保守）
                - "min": 使用组内最低bit（最激进）
                - "avg": 使用组内平均bit（四舍五入）
                
        Returns:
            统一后的bit分配
        """
        unified = dict(bit_allocation)
        
        for group_key, members in moe_groups.items():
            if not members:
                continue
            
            member_bits = [unified.get(m, 4) for m in members]
            
            if strategy == "max":
                target_bit = max(member_bits)
            elif strategy == "min":
                target_bit = min(member_bits)
            elif strategy == "avg":
                target_bit = round(sum(member_bits) / len(member_bits))
            else:
                target_bit = max(member_bits)
            
            for m in members:
                unified[m] = target_bit
        
        return unified
    
    @classmethod
    def unify_moe_layer_bits(
        cls,
        bit_allocation: Dict[str, int],
        moe_layers: Dict[str, Dict[str, List[str]]],
        strategy: str = "max",
    ) -> Dict[str, int]:
        """统一MoE层内所有专家（包括路由专家和共享专家）的bit
        
        Args:
            bit_allocation: 原始bit分配 {module_name: bits}
            moe_layers: MoE层映射 {moe_layer_key: {"experts": [...], "shared_experts": [...]}}
            strategy: 统一策略
                
        Returns:
            统一后的bit分配
        """
        unified = dict(bit_allocation)
        
        for layer_key, layer_info in moe_layers.items():
            all_members = layer_info.get("experts", []) + layer_info.get("shared_experts", [])
            
            if not all_members:
                continue
            
            member_bits = [unified.get(m, 4) for m in all_members]
            
            if strategy == "max":
                target_bit = max(member_bits)
            elif strategy == "min":
                target_bit = min(member_bits)
            elif strategy == "avg":
                target_bit = round(sum(member_bits) / len(member_bits))
            else:
                target_bit = max(member_bits)
            
            for m in all_members:
                unified[m] = target_bit
        
        return unified
    
    @classmethod
    def is_moe_container(cls, module: nn.Module) -> bool:
        """判断模块是否为MoE容器类型
        
        当模块是MoE容器时，层搜索应停止向下搜索叶子节点
        
        Args:
            module: 模块实例
            
        Returns:
            是否为MoE容器
        """
        module_type = type(module).__name__
        
        for container_type in cls.MOE_CONTAINER_TYPES:
            if container_type in module_type:
                return True
        
        if hasattr(module, "experts") and hasattr(module, "n_routed_experts"):
            return True
        
        if hasattr(module, "shared_experts"):
            return True
        
        return False
    
    @classmethod
    def get_all_moe_linear_names(cls, model: nn.Module) -> Set[str]:
        """获取所有MoE相关的Linear层名称
        
        Args:
            model: 模型实例
            
        Returns:
            MoE Linear层名称集合
        """
        moe_names = set()
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if cls.is_moe_module(name):
                    moe_names.add(name)
        
        return moe_names
