# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .moe_handler import MoEHandler


@dataclass
class LayerSelectionResult:
    """层选择结果
    
    Attributes:
        candidates: 候选层列表（包含MoE组标识和普通模块名）
        moe_groups: MoE组映射 {moe_group_key: [module_names]}
        moe_layers: MoE层映射 {moe_layer_key: {"experts": [...], "shared_experts": [...]}}
        regular_modules: 普通模块列表（非MoE）
    """
    candidates: List[str]
    moe_groups: Dict[str, List[str]]
    moe_layers: Dict[str, Dict[str, List[str]]]
    regular_modules: List[str]


class LayerSelector:
    """层选择机制
    
    支持MoE停止搜索功能，当遇到MoE容器类型时停止向下搜索叶子节点
    支持按层名称和层类型指定包含/排除规则
    """
    
    LAYER_TYPE_PATTERNS = {
        "linear": [r"\.linear\.", r"\.proj\.", r"\.fc\.", r"_proj$", r"_linear$"],
        "conv": [r"\.conv\.", r"\.conv1d\.", r"\.conv2d\.", r"_conv$"],
        "attention": [r"\.attn\.", r"\.attention\.", r"\.self_attn\.", r"_attn$"],
        "mlp": [r"\.mlp\.", r"\.feed_forward\.", r"\.ffn\.", r"_mlp$"],
        "embed": [r"\.embed\.", r"\.wte\.", r"\.wpe\.", r"embed$", r"embedding$"],
        "lm_head": [r"lm_head$", r"head$"],
        "q_proj": [r"q_proj$"],
        "k_proj": [r"k_proj$"],
        "v_proj": [r"v_proj$"],
        "o_proj": [r"o_proj$"],
        "gate_proj": [r"gate_proj$"],
        "up_proj": [r"up_proj$"],
        "down_proj": [r"down_proj$"],
    }
    
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        moe_unified: bool = True,
        moe_unify_strategy: str = "max",
        moe_unify_layer_level: bool = False,
    ):
        """初始化层选择器
        
        Args:
            include_patterns: 包含的层模式列表
            exclude_patterns: 排除的层模式列表
            moe_unified: 是否统一MoE组内bit
            moe_unify_strategy: MoE统一策略 ("max", "min", "avg")
            moe_unify_layer_level: 是否在MoE层级别统一（包含路由专家和共享专家）
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.moe_unified = moe_unified
        self.moe_unify_strategy = moe_unify_strategy
        self.moe_unify_layer_level = moe_unify_layer_level
    
    def select(self, model: nn.Module) -> LayerSelectionResult:
        """选择参与混合精度搜索的层
        
        Args:
            model: 模型实例
            
        Returns:
            LayerSelectionResult: 层选择结果
        """
        moe_groups = MoEHandler.collect_moe_modules(model)
        moe_layers = MoEHandler.collect_moe_layers(model)
        
        candidates = []
        regular_modules = []
        visited_moe_groups = set()
        visited_moe_layers = set()
        
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                continue
            
            if not self._should_include(name):
                continue
            
            moe_group = MoEHandler.get_moe_group_key(name)
            moe_layer = MoEHandler.get_moe_layer_key(name)
            
            if self.moe_unified and self.moe_unify_layer_level and moe_layer:
                if moe_layer in visited_moe_layers:
                    continue
                visited_moe_layers.add(moe_layer)
                candidates.append(moe_layer)
            elif self.moe_unified and moe_group:
                if moe_group in visited_moe_groups:
                    continue
                visited_moe_groups.add(moe_group)
                candidates.append(moe_group)
            else:
                candidates.append(name)
                regular_modules.append(name)
        
        return LayerSelectionResult(
            candidates=candidates,
            moe_groups=moe_groups,
            moe_layers=moe_layers,
            regular_modules=regular_modules,
        )
    
    def _should_include(self, name: str) -> bool:
        """检查层是否应包含在搜索中
        
        Args:
            name: 层名称
            
        Returns:
            是否应包含
        """
        for pattern in self.exclude_patterns:
            if self._match_pattern(name, pattern):
                return False
        
        if not self.include_patterns:
            return True
        
        for pattern in self.include_patterns:
            if self._match_pattern(name, pattern):
                return True
        
        return False
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """匹配层名称与模式
        
        支持以下模式类型：
        - 正则表达式（包含特殊字符）
        - 层类型别名（如 "linear", "conv"）
        - 简单子串匹配
        
        Args:
            name: 层名称
            pattern: 匹配模式
            
        Returns:
            是否匹配
        """
        if pattern in self.LAYER_TYPE_PATTERNS:
            for type_pattern in self.LAYER_TYPE_PATTERNS[pattern]:
                if re.search(type_pattern, name):
                    return True
            return False
        
        regex_chars = r"[\.\*\+\?\[\]\(\)\{\}\^\$\|\\]"
        if re.search(regex_chars, pattern):
            try:
                return bool(re.search(pattern, name))
            except re.error:
                pass
        
        return pattern in name
    
    def should_stop_search(self, module: nn.Module) -> bool:
        """判断是否应在当前模块停止搜索叶子节点
        
        当模块是MoE容器类型时返回True
        
        Args:
            module: 模块实例
            
        Returns:
            是否应停止搜索
        """
        return MoEHandler.is_moe_container(module)
    
    def get_modules_for_sensitivity(
        self,
        candidate: str,
        selection_result: LayerSelectionResult,
    ) -> List[str]:
        """获取候选层对应的所有模块名称
        
        对于MoE组，返回组内所有成员
        对于普通模块，返回单个模块
        
        Args:
            candidate: 候选层标识
            selection_result: 层选择结果
            
        Returns:
            模块名称列表
        """
        if candidate in selection_result.moe_groups:
            return selection_result.moe_groups[candidate]
        
        if candidate in selection_result.moe_layers:
            layer_info = selection_result.moe_layers[candidate]
            return layer_info.get("experts", []) + layer_info.get("shared_experts", [])
        
        return [candidate]
    
    def expand_candidates_to_modules(
        self,
        candidates: List[str],
        selection_result: LayerSelectionResult,
    ) -> List[str]:
        """将候选层列表展开为模块名称列表
        
        Args:
            candidates: 候选层列表
            selection_result: 层选择结果
            
        Returns:
            模块名称列表
        """
        modules = []
        for candidate in candidates:
            modules.extend(self.get_modules_for_sensitivity(candidate, selection_result))
        return modules
