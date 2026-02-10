"""Activation data collection for DiceQ rotation matrix training."""

import argparse
import functools
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from ...utils.logger import setup_logger
from .data_utils import get_loaders

# 初始化 logger
logger = setup_logger()

default_path = "../data/train_data/"


def _resolve_runtime_device(args) -> str:
    device = getattr(args, "device", None)
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, str) and device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_input_from_batch(batch: Any, device: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(batch, dict):
        prepared = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(device)
        if "input_ids" in prepared:
            input_ids = prepared.pop("input_ids")
        elif prepared:
            first_key = next(iter(prepared))
            input_ids = prepared.pop(first_key)
        else:
            raise ValueError("批数据字典不包含可用的 tensor 字段。")
        return input_ids, prepared

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("批数据为空。")
        first = batch[0]
        if not torch.is_tensor(first):
            raise ValueError(f"不支持的批数据首元素类型: {type(first)}")
        return first.to(device), {}

    if torch.is_tensor(batch):
        return batch.to(device), {}

    raise ValueError(f"不支持的批数据类型: {type(batch)}")


@dataclass
class LayerSelector:
    """
    灵活的神经网络层选择器，支持多种方式选择目标层。
    
    支持的选择方式：
    - 精确名称匹配: name='model.layers.0.self_attn.q_proj'
    - 模式匹配（子字符串）: pattern='self_attn' 匹配所有包含'self_attn'的层
    - 层类型匹配: layer_type=nn.Linear 匹配所有Linear层
    - 索引匹配: index=0 匹配指定索引的层
    
    Attributes:
        name: 精确匹配的模块名称
        layer_type: 模块类型匹配（如 nn.Linear）
        index: 层索引匹配
        pattern: 子字符串模式匹配
    """
    name: Optional[str] = None
    layer_type: Optional[type] = None
    index: Optional[int] = None
    pattern: Optional[str] = None
    
    def matches(self, module_name: str, module: nn.Module, layer_idx: Optional[int] = None) -> bool:
        """
        检查模块是否匹配选择条件。
        
        Args:
            module_name: 模块的完整名称
            module: 模块实例
            layer_idx: 可选的层索引
            
        Returns:
            如果模块匹配任何选择条件则返回True
        """
        if self.name is not None and module_name == self.name:
            return True
        if self.pattern is not None and self.pattern in module_name:
            return True
        if self.layer_type is not None and isinstance(module, self.layer_type):
            return True
        if self.index is not None and layer_idx is not None and layer_idx == self.index:
            return True
        return False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LayerSelector':
        """
        从配置字典创建LayerSelector实例。
        
        Args:
            config: 配置字典，可以包含name、pattern、layer_type、index等键
            
        Returns:
            LayerSelector实例
        """
        if 'layer_type' in config and isinstance(config['layer_type'], str):
            type_map = {
                'Linear': nn.Linear,
                'Conv2d': nn.Conv2d,
                'Conv1d': nn.Conv1d,
                'BatchNorm2d': nn.BatchNorm2d,
                'LayerNorm': nn.LayerNorm,
            }
            config['layer_type'] = type_map.get(config['layer_type'], nn.Linear)
        return cls(**config)


class ActivationHookManager:
    """
    管理神经网络层的钩子注册和激活数据收集。
    
    提供灵活接口用于：
    - 在任意层注册前向钩子
    - 捕获输入/输出激活
    - 按层名组织捕获的数据
    - 保存捕获的数据到磁盘
    - 使用后清理钩子
    
    Example:
        >>> hook_manager = ActivationHookManager()
        >>> selectors = [LayerSelector(pattern='self_attn.q_proj')]
        >>> hook_manager.register_hooks(model, selectors, capture_input=True)
        >>> # 运行推理...
        >>> data = hook_manager.get_captured_data()
        >>> hook_manager.clear_hooks()
    """
    def __init__(self):
        self.hooks: List[Any] = []
        self.captured_data: Dict[str, List[torch.Tensor]] = {}
        self.layer_indices: Dict[str, int] = {}
        self.matched_layers: List[str] = []  # 记录匹配到的层名称
        
    def register_hooks(self, 
                      model: nn.Module,
                      layer_selectors: List[LayerSelector],
                      hook_func: Optional[Callable] = None,
                      capture_input: bool = True,
                      capture_output: bool = False,
                      validate: bool = True) -> List[str]:
        """
        为指定层注册钩子。
        
        Args:
            model: 要注册钩子的模型
            layer_selectors: 层选择器列表
            hook_func: 可选的自定义钩子函数
            capture_input: 是否捕获输入激活
            capture_output: 是否捕获输出激活
            validate: 是否验证至少匹配到一个层
            
        Returns:
            匹配到的层名称列表
            
        Raises:
            ValueError: 如果validate为True且没有匹配到任何层
        """
        self.clear_hooks()
        self.captured_data.clear()
        self.layer_indices.clear()
        self.matched_layers.clear()
        
        for layer_idx, (name, module) in enumerate(model.named_modules()):
            matched = False
            for selector in layer_selectors:
                if selector.matches(name, module, layer_idx):
                    matched = True
                    break
            
            if matched:
                self._register_single_hook(module, name, layer_idx, hook_func, capture_input, capture_output)
                self.matched_layers.append(name)
        
        if validate and not self.matched_layers:
            selector_info = [f"pattern={s.pattern}, name={s.name}, type={s.layer_type}, index={s.index}" 
                           for s in layer_selectors]
            raise ValueError(
                f"未找到匹配任何选择条件的层。\n"
                f"选择器配置: {selector_info}\n"
                f"模型中的层: {list(dict(model.named_modules()).keys())[:20]}..."
            )
        
        return self.matched_layers
    
    def _register_single_hook(self,
                              module: nn.Module,
                              name: str,
                              layer_idx: int,
                              hook_func: Optional[Callable],
                              capture_input: bool,
                              capture_output: bool) -> None:
        """
        为单个模块注册钩子。
        """
        def default_hook(module, input, output, hook_name, hook_layer_idx):
            if capture_input:
                input_tensor = input[0] if isinstance(input, tuple) else input
                if hook_name not in self.captured_data:
                    self.captured_data[hook_name] = []
                    self.layer_indices[hook_name] = hook_layer_idx
                self.captured_data[hook_name].append(input_tensor.detach().cpu())
            
            if capture_output:
                output_tensor = output[0] if isinstance(output, tuple) else output
                output_key = f"{hook_name}_output"
                if output_key not in self.captured_data:
                    self.captured_data[output_key] = []
                    self.layer_indices[output_key] = hook_layer_idx
                self.captured_data[output_key].append(output_tensor.detach().cpu())
            
            if hook_func is not None:
                hook_func(module, input, output, hook_name, hook_layer_idx)
        
        hook = module.register_forward_hook(
            functools.partial(default_hook, hook_name=name, hook_layer_idx=layer_idx)
        )
        self.hooks.append(hook)
    
    def clear_hooks(self) -> None:
        """
        清除所有已注册的钩子。
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_captured_data(self, layer_name: Optional[str] = None) -> Union[Dict[str, List[torch.Tensor]], List[torch.Tensor]]:
        """
        获取捕获的激活数据。
        
        Args:
            layer_name: 可选的特定层名称
            
        Returns:
            如果指定layer_name则返回该层的数据列表，否则返回所有数据字典
        """
        if layer_name is not None:
            if layer_name not in self.captured_data:
                raise ValueError(f"未找到层的捕获数据: {layer_name}")
            return self.captured_data[layer_name]
        return self.captured_data
    
    def get_layer_index(self, layer_name: str) -> int:
        """
        获取层的索引。
        """
        if layer_name not in self.layer_indices:
            raise ValueError(f"未找到层: {layer_name}")
        return self.layer_indices[layer_name]
    
    def get_matched_layers(self) -> List[str]:
        """
        获取所有匹配到的层名称列表。
        """
        return self.matched_layers.copy()
    
    def save_captured_data(self, save_path: str, layer_name: Optional[str] = None) -> None:
        """
        保存捕获的数据到磁盘。
        """
        os.makedirs(save_path, exist_ok=True)
        data_to_save = self.captured_data if layer_name is None else {layer_name: self.captured_data.get(layer_name)}
        
        for name, tensors in data_to_save.items():
            layer_idx = self.layer_indices.get(name, -1)
            for idx, tensor in enumerate(tensors):
                filename = f"sample_{idx}_layer_{layer_idx}_{name.replace('.', '_')}.pt"
                filepath = os.path.join(save_path, filename)
                torch.save(tensor, filepath)


class ActivationDataCollector:
    """
    从神经网络层收集激活数据的高级接口。
    
    通过以下方式简化激活数据收集过程：
    - 支持完全自定义的层配置
    - 支持动态R类型配置
    - 自动处理数据收集和存储
    - 管理钩子生命周期
    
    Attributes:
        model: 神经网络模型
        args: 配置参数
        hook_manager: 内部钩子管理器实例
        layer_config: 指定从哪些层收集数据的配置
    """
    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        self.hook_manager = ActivationHookManager()
        self.layer_config = {}  # 动态配置，不设置默认值
        self.save_mapping = {}  # 动态配置，不设置默认值
        self.sampling_config = {}  # 采样配置
        self.r_types = []  # 动态R类型列表
        
    def _apply_config(self, config: Dict[str, Any], source_name: str = "dict") -> None:
        self.layer_config = {}
        self.save_mapping = {}
        self.sampling_config = {}
        self.r_types = []

        for r_type, r_config in config.items():
            self.r_types.append(r_type)

            folder = r_config.get('folder', f"{r_type}")
            save_mode = r_config.get('save_mode', 'separate')
            filename_template = r_config.get(
                'filename_template',
                'sample_{idx}_layer_{layer_idx}_{name}.pt'
            )
            sampling = r_config.get('sampling', {})

            # layers 是字符串数组，每个字符串是一个 pattern
            layers_list = r_config.get('layers', [])
            if not isinstance(layers_list, list):
                raise ValueError(f"配置错误: {r_type}.layers 必须是字符串数组")

            # 为每个 pattern 创建 layer_config 条目
            for i, pattern in enumerate(layers_list):
                layer_name = f"{r_type}_layer_{i}"
                self.layer_config[layer_name] = {
                    'pattern': pattern,
                    'data_type': r_type,  # 所有 layer 指向同一个 r_type
                    'r_type': r_type
                }

            # 为 r_type 创建统一的保存映射
            self.save_mapping[r_type] = {
                'path_key': r_type,
                'save_mode': save_mode,
                'filename_template': filename_template,
                'folder': folder,
                'layers': layers_list  # 保存所有 patterns 用于后续处理
            }

            if sampling:
                self.sampling_config[r_type] = sampling

        logger.info(f"已从 {source_name} 加载配置:")
        logger.info(f"  - R类型: {self.r_types}")
        logger.info(f"  - 层配置: {self.layer_config}")
        logger.info(f"  - 保存映射: {self.save_mapping}")
        logger.info(f"  - 采样配置: {self.sampling_config}")

    def configure_from_file(self, config_path: str) -> None:
        """
            从配置文件加载所有配置。

            配置文件格式:
            {
                "r_type_name": {
                    "layers": ["pattern1", "pattern2"],  // 字符串数组
                    "save_mode": "separate" | "combined",
                    "filename_template": "...",
                    "folder": "...",
                    "sampling": {"max_tokens": 2048}  // 可选
                }
            }

            Args:
                config_path: 配置文件路径（JSON格式）

            Raises:
                FileNotFoundError: 配置文件不存在
                json.JSONDecodeError: 配置文件格式错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self._apply_config(config=config, source_name=config_path)

    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict) or not config:
            raise ValueError("配置对象必须是非空字典。")
        self._apply_config(config=config, source_name="in-memory config")
    
    def configure_from_args(self, 
                           r_types: List[str],
                           layer_config: Dict[str, List[str]],
                           save_mapping: Dict[str, Dict[str, Any]],
                           sampling_config: Optional[Dict[str, Any]] = None) -> None:
        """
        从命令行参数配置收集器。
        
        Args:
            r_types: R类型列表，如 ['r1', 'r2', 'r4']
            layer_config: 层配置字典
            save_mapping: 保存映射字典
            sampling_config: 可选的采样配置
        """
        self.r_types = r_types
        self.layer_config = layer_config
        self.save_mapping = save_mapping
        self.sampling_config = sampling_config or {}
        
        logger.info(f"已从命令行参数配置:")
        logger.info(f"  - R类型: {self.r_types}")
        logger.info(f"  - 层配置: {self.layer_config}")
        logger.info(f"  - 保存映射: {self.save_mapping}")
        logger.info(f"  - 采样配置: {self.sampling_config}")
    
    def get_r_types(self) -> List[str]:
        """
        获取当前配置的R类型列表。
        """
        return self.r_types.copy()
    
    def _create_layer_selectors(self, data_type: str, module_names: List[str]) -> List[LayerSelector]:
        """
        为指定数据类型创建层选择器列表。
        """
        selectors = []
        for module_name in module_names:
            selector = LayerSelector(pattern=module_name)
            selectors.append(selector)
        return selectors
    
    def validate_layer_config(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        验证层配置是否能匹配到模型中的层。

        按 r_type 验证，每个 r_type 可以包含多个 layer pattern。

        Args:
            model: 要验证的模型

        Returns:
            匹配结果字典，键为 r_type，值为匹配到的层名称列表

        Raises:
            ValueError: 如果某个 r_type 没有匹配到任何层
        """
        validation_results = {}
        all_module_names = [name for name, _ in model.named_modules()]

        for r_type in self.r_types:
            mapping = self.save_mapping.get(r_type, {})
            layer_patterns = mapping.get('layers', [])

            matched_layers = []
            for pattern in layer_patterns:
                matching = [name for name in all_module_names if pattern in name]
                matched_layers.extend(matching)

            validation_results[r_type] = matched_layers

            if not matched_layers:
                raise ValueError(
                    f"r_type '{r_type}' 的配置 {layer_patterns} 未匹配到任何层。\n"
                    f"模型中的层示例: {all_module_names[:20]}..."
                )

        return validation_results
    
    @staticmethod
    def _normalize_separate_sample(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    def _save_r_type_data(self,
                          r_type: str,
                          all_tensors: List[torch.Tensor],
                          layer_idx: int,
                          save_paths: Dict[str, str],
                          name_override: Optional[str] = None) -> None:
        """
        保存单个 r_type 的数据到指定路径。

        Args:
            r_type: R类型名称
            all_tensors: 要保存的tensor列表
            layer_idx: 层索引
            save_paths: 保存路径字典
        """
        if r_type not in self.save_mapping:
            logger.warning(f"数据类型 '{r_type}' 没有配置保存映射，跳过保存")
            return

        mapping = self.save_mapping[r_type]
        path_key = mapping['path_key']
        save_mode = mapping['save_mode']
        filename_template = mapping['filename_template']
        save_name = name_override if name_override is not None else r_type

        if path_key not in save_paths:
            logger.warning(f"保存路径键 '{path_key}' 不存在于 save_paths 中，跳过 '{r_type}'")
            return

        path = save_paths[path_key]
        os.makedirs(path, exist_ok=True)

        if save_mode == 'combined':
            # 合并所有样本
            combined = torch.cat(all_tensors, dim=0)
            # 检查是否需要采样
            sampling = self.sampling_config.get(r_type, {})
            max_tokens = sampling.get('max_tokens', -1)
            if max_tokens > 0 and combined.shape[1] > max_tokens:
                token_idx = torch.randperm(combined.shape[1])[:max_tokens]
                combined = combined[:, token_idx, :]
            filename = filename_template.format(
                idx=0,
                layer_idx=layer_idx,
                name=save_name
            )
            torch.save(combined, os.path.join(path, filename))
        elif save_mode == 'separate':
            # 每个样本单独保存
            for idx, tensor in enumerate(all_tensors):
                tensor = self._normalize_separate_sample(tensor)
                filename = filename_template.format(
                    idx=idx,
                    layer_idx=layer_idx,
                    name=save_name
                )
                torch.save(tensor, os.path.join(path, filename))
        else:
            logger.warning(f"未知的保存模式 '{save_mode}'，跳过 '{r_type}'")

    def collect_layer_activations(self,
                                  decoder_layer: nn.Module,
                                  layer_idx: int,
                                  input_data: torch.Tensor,
                                  save_paths: Dict[str, str],
                                  **kwargs) -> torch.Tensor:
        """
        从指定层收集激活数据并立即保存。

        所有 r_type 共享同一次前向传播，确保输入一致性。

        Args:
            decoder_layer: 解码器层模块
            layer_idx: 层索引
            input_data: 输入数据
            save_paths: 保存路径字典
            **kwargs: 所有前向传播参数（包括 attention_mask, position_ids 等）

        Returns:
            torch.Tensor: 更新后的 input_data
        """
        # 直接使用传入的 kwargs 作为前向传播参数
        forward_kwargs = kwargs

        # 1. 收集所有 r_type 的所有 selectors
        all_selectors = []
        r_type_patterns = {}  # 记录每个 r_type 对应的 patterns

        for r_type in self.r_types:
            mapping = self.save_mapping.get(r_type, {})
            layer_patterns = mapping.get('layers', [])

            if not layer_patterns:
                continue

            selectors = []
            for pattern in layer_patterns:
                selector = LayerSelector(pattern=pattern)
                selectors.append(selector)
                all_selectors.append(selector)

            r_type_patterns[r_type] = {
                'selectors': selectors,
                'patterns': layer_patterns,
                'mapping': mapping
            }

        if not all_selectors:
            logger.warning(f"层 {layer_idx} 没有配置任何 r_type，跳过")
            return input_data

        # 2. 一次性注册所有钩子（所有 r_type）
        try:
            matched = self.hook_manager.register_hooks(
                decoder_layer,
                all_selectors,
                capture_input=True,
                capture_output=False,
                validate=True
            )
            logger.info(f"层 {layer_idx} 成功匹配到层: {matched}")
        except ValueError as e:
            logger.warning(f"层 {layer_idx} 注册钩子失败: {str(e)}")
            return input_data

        # 3. 执行一次前向传播（所有 r_type 共享）
        with torch.no_grad():
            for j in range(self.args.nsamples):
                input_data[j] = decoder_layer(
                    input_data[j].unsqueeze(0),
                    **forward_kwargs
                )[0]

        # 4. 获取所有捕获的数据
        captured = self.hook_manager.get_captured_data()
        self.hook_manager.clear_hooks()

        if not captured:
            logger.warning(f"层 {layer_idx} 未收集到任何数据")
            return input_data

        # 5. 按 r_type 分组数据并分别保存
        for r_type, config in r_type_patterns.items():
            patterns = config['patterns']
            mapping = config['mapping']
            save_mode = mapping.get('save_mode', 'separate')

            # 从 captured 中提取属于当前 r_type 的数据（按 module 分组）
            module_tensors: Dict[str, List[torch.Tensor]] = {}
            for module_name, tensors in captured.items():
                for pattern in patterns:
                    if pattern in module_name:
                        module_tensors.setdefault(module_name, []).extend(tensors)
                        break

            if not module_tensors:
                logger.warning(f"层 {layer_idx} 的 r_type '{r_type}' 未收集到任何数据")
                continue

            for module_name, tensors in module_tensors.items():
                if len(tensors) != self.args.nsamples:
                    raise ValueError(
                        f"期望 {self.args.nsamples} 个样本，"
                        f"但 {module_name} 得到 {len(tensors)} 个"
                    )

            if save_mode == 'separate':
                for module_name, tensors in module_tensors.items():
                    save_name = module_name.replace('.', '_')
                    logger.info(
                        f"层 {layer_idx} 的 r_type '{r_type}' 模块 '{module_name}' 收集到 "
                        f"{len(tensors)} 个张量"
                    )
                    self._save_r_type_data(
                        r_type,
                        tensors,
                        layer_idx,
                        save_paths,
                        name_override=save_name,
                    )
            else:
                merged_tensors: List[torch.Tensor] = []
                for tensors in module_tensors.values():
                    merged_tensors.extend(tensors)
                logger.info(
                    f"层 {layer_idx} 的 r_type '{r_type}' 收集到 "
                    f"{len(merged_tensors)} 个张量"
                )
                self._save_r_type_data(r_type, merged_tensors, layer_idx, save_paths)

        # 6. 释放内存
        del captured
        torch.cuda.empty_cache()

        return input_data
    
    def save_collected_data(self, 
                            collected_data: Dict[str, torch.Tensor],
                            layer_idx: int,
                            save_paths: Dict[str, str]) -> None:
        """
        保存收集的数据到指定路径。
        
        使用 self.save_mapping 配置决定每个数据类型保存到哪个路径。
        """
        for data_type, data in collected_data.items():
            if data_type not in self.save_mapping:
                logger.warning(f"数据类型 '{data_type}' 没有配置保存映射，跳过保存")
                continue

            mapping = self.save_mapping[data_type]
            path_key = mapping['path_key']
            save_mode = mapping['save_mode']
            filename_template = mapping['filename_template']

            if path_key not in save_paths:
                logger.warning(f"保存路径键 '{path_key}' 不存在于 save_paths 中，跳过 '{data_type}'")
                continue

            path = save_paths[path_key]
            os.makedirs(path, exist_ok=True)

            if save_mode == 'separate':
                # 每个样本单独保存
                for idx, tensor in enumerate(data):
                    filename = filename_template.format(
                        idx=idx,
                        layer_idx=layer_idx,
                        name=data_type
                    )
                    torch.save(tensor, os.path.join(path, filename))
            elif save_mode == 'combined':
                # 合并保存
                filename = filename_template.format(
                    idx=0,
                    layer_idx=layer_idx,
                    name=data_type
                )
                torch.save(data, os.path.join(path, filename))
            else:
                logger.warning(f"未知的保存模式 '{save_mode}'，跳过 '{data_type}'")


def get_Rs_training_data(model, dataloader, save_path, args):
    """
    使用基于钩子的激活收集方式收集训练数据。
    
    此函数从神经网络模型的指定层收集激活数据，用于量化校准。
    使用灵活的基于钩子的机制，支持动态配置的R类型。
    
    函数执行流程：
    1. 使用Catcher模块捕获第一层输入
    2. 遍历每个解码器层
    3. 使用ActivationDataCollector从配置的层收集激活
    4. 将收集的数据保存到指定路径
    
    Args:
        model: 要收集数据的神经网络模型
        dataloader: 提供校准数据的DataLoader
        save_path: 将R类型映射到保存路径的字典
        args: 配置参数，包括：
            - nsamples: 要收集的样本数
            - seqlen: 序列长度
            - r_config_file: R类型配置文件路径
            
    Returns:
        恢复use_cache配置的模型
        
    Raises:
        ValueError: 如果模型类型不受支持或激活收集失败
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    config = model.config
    dtype = config.torch_dtype if config.torch_dtype is not None else torch.float32
    dev = _resolve_runtime_device(args)
    nsamples = int(getattr(args, "nsamples", 128))
    seqlen = int(getattr(args, "seqlen", 2048))

    try:
        if hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        layers = model.model.layers

        layers[0] = layers[0].to(dev)
        inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
        cache = {"i": 0}

        # 捕获第一层输入
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *margs, **kwargs):
                if len(margs) > 0 and cache["i"] < nsamples:
                    captured = margs[0]
                    if captured.ndim == 3 and captured.shape[0] == 1:
                        captured = captured.squeeze(0)
                    if captured.shape[0] != seqlen:
                        if captured.shape[0] > seqlen:
                            captured = captured[:seqlen, :]
                        else:
                            padded = torch.zeros((seqlen, captured.shape[-1]), dtype=captured.dtype, device=captured.device)
                            padded[:captured.shape[0], :] = captured
                            captured = padded
                    inps[cache["i"]] = captured
                    cache["i"] += 1
                if cache["i"] == 1 and "kwargs" not in cache:
                    cache["kwargs"] = dict(kwargs)
                raise ValueError

        layers[0] = Catcher(layers[0])
        with torch.no_grad():
            for batch in dataloader:
                if cache["i"] >= nsamples:
                    break
                try:
                    input_ids, batch_kwargs = _extract_input_from_batch(batch, dev)
                    model(input_ids, **batch_kwargs)
                except ValueError:
                    pass

        # 将嵌入层和第一层移到CPU
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        if hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.cpu()

        actual_samples = cache["i"]
        if actual_samples == 0:
            raise ValueError("未捕获到任何可用样本，无法进行 DiceQ 训练数据收集。")

        fp_inps = inps[:actual_samples]
        forward_kwargs = dict(cache.get("kwargs", {}))
        del inps, cache

        data_collector = ActivationDataCollector(model, args)

        r_config_dict = getattr(args, "r_config", None)
        if isinstance(r_config_dict, dict) and r_config_dict:
            data_collector.configure_from_dict(r_config_dict)
        else:
            if not hasattr(args, 'r_config_file') or not args.r_config_file:
                raise ValueError("必须通过 --r_config_file 指定配置文件或通过 r_config 传入配置对象！")
            data_collector.configure_from_file(args.r_config_file)

        try:
            validation_results = data_collector.validate_layer_config(layers[0])
            logger.info(f"层配置验证通过，匹配结果: {validation_results}")
        except ValueError as e:
            raise ValueError(f"层配置验证失败: {str(e)}")

        for idx, decoder_layer in tqdm(enumerate(layers), total=len(layers), desc="处理层"):
            decoder_layer = decoder_layer.to(dev)

            try:
                # 流式收集和保存：每个 r_type 收集完立即保存，不缓存所有数据
                fp_inps = data_collector.collect_layer_activations(
                    decoder_layer=decoder_layer,
                    layer_idx=idx,
                    input_data=fp_inps,
                    save_paths=save_path,
                    **forward_kwargs
                )
            except ValueError as e:
                raise ValueError(f"收集层 {idx} 的激活时出错: {str(e)}")

            decoder_layer = decoder_layer.to('cpu')

        del fp_inps
        torch.cuda.empty_cache()
        gc.collect()
        return model
    finally:
        model.config.use_cache = use_cache


def collect_training_data_for_diceq(
    model: nn.Module,
    dataloader: Any,
    output_root: Union[str, Path],
    r_config: Optional[Dict[str, Any]] = None,
    r_config_file: Optional[Union[str, Path]] = None,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    device: str = "auto",
) -> Dict[str, str]:
    """
    Pipeline接口：按配置收集 DiceQ 训练数据并返回每个 target 的数据目录。
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    r_config_path = None
    if r_config is None:
        if r_config_file is None:
            raise ValueError("必须提供 r_config 或 r_config_file。")
        r_config_path = Path(r_config_file)
        if not r_config_path.exists():
            raise FileNotFoundError(f"DiceQ 配置文件不存在: {r_config_path}")
        with r_config_path.open("r", encoding="utf-8") as f:
            r_config = json.load(f)
    if not isinstance(r_config, dict) or not r_config:
        raise ValueError("DiceQ 配置文件必须是非空 JSON 对象。")

    save_path: Dict[str, str] = {}
    for r_type, cfg in r_config.items():
        folder = cfg.get("folder", r_type) if isinstance(cfg, dict) else r_type
        path = output_root / folder
        path.mkdir(parents=True, exist_ok=True)
        save_path[r_type] = str(path)

    args = argparse.Namespace(
        nsamples=int(nsamples),
        seqlen=int(seqlen),
        r_config_file=str(r_config_path) if r_config_path else None,
        r_config=r_config,
        seed=int(seed),
        device=device,
    )
    get_Rs_training_data(model=model, dataloader=dataloader, save_path=save_path, args=args)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/data02/datasets/GLM-4.7-Flash',
                        help='模型名称或路径')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace token')
    parser.add_argument('--r_path', type=str, default='output/DICEQuant/data',
                        help='保存训练数据的根路径')
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4"],
                        help="校准数据来源。")
    parser.add_argument('--nsamples', type=int, default=128,
                        help='样本数量')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='序列长度')
    parser.add_argument("--seed", type=int, default=0, 
                        help="校准数据采样的随机种子。")
    
    # 配置文件参数（强制要求）
    parser.add_argument('--r_config_file', type=str, default='config/layer_config_example.json',
                        help='R类型配置文件路径（JSON格式），定义R类型、层配置和保存映射')
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    dataset_info = f"{args.calib_dataset}_{args.nsamples}samples"
    
    # 检查配置文件是否存在
    if not os.path.exists(args.r_config_file):
        raise FileNotFoundError(f"配置文件不存在: {args.r_config_file}")
    
    # 从配置文件加载 R 类型配置
    with open(args.r_config_file, 'r', encoding='utf-8') as f:
        r_config = json.load(f)
    
    # 确定根路径
    base_path =  args.r_path
    
    # 根据配置文件创建路径
    args.r_path = {}
    for r_type, config in r_config.items():
        r_folder = config.get('folder', f"{r_type}")
        args.r_path[r_type] = os.path.join(
            base_path,
            dataset_info,
            model_name,
            r_folder
        )

    # 创建所有路径
    for r_type, path in args.r_path.items():
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info(f"---> {r_type}_path: {path}")

    return args


@torch.no_grad()
def main():
    """Standalone entry point for activation data collection."""
    args = parse_args()
    transformers.set_seed(args.seed)
    
    # Load model using transformers directly
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token,
    )

    dataloader = get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        eval_mode=False
    )
    get_Rs_training_data(
        model=model,
        dataloader=dataloader,
        save_path=args.r_path,
        args=args
    )


if __name__ == "__main__":
    main()
