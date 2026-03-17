# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import re
import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
import torch.nn as nn

from .moe_handler import MoEHandler


@dataclass
class LayerSelectionResult:
    """层选择结果
    
    Attributes:
        candidates: 候选层列表（包含组标识和模块名）
        moe_groups: MoE组映射 {moe_group_key: [module_names]}
        moe_layers: MoE层映射 {moe_layer_key: {"experts": [...], "shared_experts": [...]}}
        regular_modules: 普通模块列表（非MoE）
        fixed_bits: 固定bit配置的层 {module_name: bits or None}
        group_mapping: 分组映射 {module_name: group_name}
    """
    candidates: List[str]
    moe_groups: Dict[str, List[str]]
    moe_layers: Dict[str, Dict[str, List[str]]]
    regular_modules: List[str]
    fixed_bits: Dict[str, Optional[int]] = field(default_factory=dict)
    group_mapping: Dict[str, str] = field(default_factory=dict)


class LayerSelector:
    """层选择机制
    
    支持功能：
    - MoE停止搜索：遇到MoE容器类型时停止向下搜索叶子节点
    - 包含/排除规则：按层名称和层类型指定包含/排除规则
    - 灵活分组：按配置路径将层分组
    - 固定bit：指定某些层的固定量化bit（None表示不量化）
    
    MoE behavior:
    - moe_unified=True: 每个MoE层的路由专家统一为一个组
    - Shared experts: Always independent (each linear gets its own bit)
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
        group_by_path: Optional[Dict[str, str]] = None,
        fixed_bits: Optional[Dict[str, Optional[int]]] = None,
        moe_unified: bool = True,
    ):
        """初始化层选择器
        
        Args:
            include_patterns: 包含的层模式列表
            exclude_patterns: 排除的层模式列表
            group_by_path: 分组配置 {"path_pattern": "group_name"}
            fixed_bits: 固定bit配置 {"module_path": bits}，设为None表示不量化
            moe_unified: 是否将每个MoE层的路由专家统一为一个组
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.group_by_path = group_by_path or {}
        self.fixed_bits = fixed_bits or {}
        self.moe_unified = moe_unified
    
    def _match_path_pattern(self, name: str, pattern: str) -> bool:
        """匹配路径模式，支持通配符
        
        Args:
            name: 模块名称
            pattern: 模式（支持 * 通配符）
            
        Returns:
            是否匹配
        """
        return fnmatch.fnmatch(name, pattern)
    
    def _get_fixed_bits_for_module(self, module_name: str) -> tuple[bool, Optional[int]]:
        """获取模块的固定bit配置

        优先精确匹配，其次通配符匹配

        Args:
            module_name: 模块名称

        Returns:
            (是否匹配, 固定bit值)
        """
        if module_name in self.fixed_bits:
            return True, self.fixed_bits[module_name]

        for pattern, bits in self.fixed_bits.items():
            if self._match_path_pattern(module_name, pattern):
                return True, bits

        return False, None
    
    def _get_group_for_module(self, module_name: str) -> Optional[str]:
        """获取模块的分组名称
        
        Args:
            module_name: 模块名称
            
        Returns:
            分组名称，如果没有配置则返回None
        """
        for pattern, group_name in self.group_by_path.items():
            if self._match_path_pattern(module_name, pattern):
                return group_name
        return None
    
    def select(self, model: nn.Module) -> LayerSelectionResult:
        """选择参与混合精度搜索的层
        
        流程：
        1. 遍历模型所有Linear层
        2. 应用include/exclude过滤
        3. 检查MoE路由层排除规则
        4. 检查固定bit配置
        5. 应用group_by_path分组
        6. MoE路由专家统一分组
        
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
        visited_groups = set()
        fixed_bits_map = {}
        group_mapping = {}
        
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                continue
            
            if not self._should_include(name):
                continue
            
            fixed_bits_matched, fixed_bits = self._get_fixed_bits_for_module(name)
            if fixed_bits_matched:
                fixed_bits_map[name] = fixed_bits
                continue
            
            if MoEHandler.is_shared_expert(name):
                candidates.append(name)
                regular_modules.append(name)
                continue
            
            if MoEHandler.is_routed_expert(name):
                if not MoEHandler.is_routed_expert_projection(name):
                    continue

                if self.moe_unified:
                    moe_group = MoEHandler.get_moe_group_key(name)
                    if not moe_group:
                        continue
                    if moe_group in visited_moe_groups:
                        continue
                    visited_moe_groups.add(moe_group)
                    candidates.append(moe_group)
                else:
                    candidates.append(name)
                    regular_modules.append(name)
                continue
            
            group_name = self._get_group_for_module(name)
            if group_name:
                if group_name not in visited_groups:
                    visited_groups.add(group_name)
                    candidates.append(group_name)
                    group_mapping[name] = group_name
                else:
                    group_mapping[name] = group_name
            else:
                candidates.append(name)
                regular_modules.append(name)
        
        return LayerSelectionResult(
            candidates=candidates,
            moe_groups=moe_groups,
            moe_layers=moe_layers,
            regular_modules=regular_modules,
            fixed_bits=fixed_bits_map,
            group_mapping=group_mapping,
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
        对于分组，返回组内所有成员
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
        
        modules_in_group = [k for k, v in selection_result.group_mapping.items() if v == candidate]
        if modules_in_group:
            return modules_in_group
        
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
