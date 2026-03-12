# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.logger import setup_logger
from .mixsearch_config import validate_mix_precision_config
from .layer_selector import LayerSelector, LayerSelectionResult
from .metrics import MetricFactory
from .moe_handler import MoEHandler

log = setup_logger()


@dataclass
class SearchResult:
    """搜索结果
    
    Attributes:
        bit_allocation: 模块名到比特配置的映射
        sensitivities: 模块名到敏感度的映射
        statistics: 统计信息
        fixed_bits: 固定bit配置的层
    """
    bit_allocation: Dict[str, Dict[str, int]]
    sensitivities: Dict[str, float]
    statistics: Dict[str, Any]
    fixed_bits: Dict[str, int]


class MixPrecisionSearchEngine:
    """混合精度搜索引擎
    
    执行混合精度权重搜索，为每层分配合适的比特数
    仅针对权重进行bit搜索，不包含激活量化
    
    Features:
    - Weight-only bit search
    - Flexible layer grouping via group_by_path
    - Fixed bit configuration
    - MoE routing expert unification
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        calibration_data: List,
        device: Optional[torch.device] = None,
    ):
        """初始化搜索引擎
        
        Args:
            model: 模型实例
            config: 混合精度配置字典
            calibration_data: 校准数据
            device: 执行设备
        """
        self.model = model
        self.config = config
        self.calibration_data = calibration_data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        validate_mix_precision_config(config)
        
        self.layer_selector = LayerSelector(
            include_patterns=config.get("layer_include"),
            exclude_patterns=config.get("layer_exclude"),
            group_by_path=config.get("group_by_path"),
            fixed_bits=config.get("fixed_bits"),
            moe_unified=config.get("moe_unified", True),
        )
        self.metric = MetricFactory.create(config.get("metric", "kl"))
        self.metric_name = str(config.get("metric", "kl")).lower()
        
        self.weight_bits = config.get("weight_bits", [4, 8])
        self.policy = config.get("policy", "topk")
        self.topk = config.get("topk", 0.1)
        self.threshold = config.get("threshold", 1e-6)
        self.moe_unified = config.get("moe_unified", True)
        self.output_reduction = config.get("output_reduction", "last_token")
        self.store_baseline_on_cpu = bool(config.get("store_baseline_on_cpu", True))

        max_batches = config.get("max_calibration_batches")
        if isinstance(max_batches, int) and max_batches > 0:
            self.calibration_data = list(islice(self.calibration_data, max_batches))
            log.info(
                "Mix precision search: limiting calibration batches to %s",
                len(self.calibration_data),
            )
        
        self._baseline_outputs = {}
    
    def search(self) -> SearchResult:
        """执行混合精度搜索
        
        Returns:
            SearchResult: 搜索结果
        """
        log.info("Starting mix precision search...")
        
        selection_result = self.layer_selector.select(self.model)
        candidates = selection_result.candidates
        
        log.info(f"Found {len(candidates)} candidates for mix precision search")
        log.info(f"MoE groups: {len(selection_result.moe_groups)}")
        log.info(f"Fixed bits layers: {len(selection_result.fixed_bits)}")
        log.info("Sensitivity metric: %s", self._describe_metric())
        
        self._run_baseline()
        
        sensitivities = self._compute_sensitivities(candidates, selection_result)
        
        bit_allocation = self._allocate_bits(sensitivities, selection_result)
        
        statistics = self._compute_statistics(bit_allocation, sensitivities)
        
        log.info("Mix precision search completed.")
        log.info(f"Statistics: {statistics}")
        
        return SearchResult(
            bit_allocation=bit_allocation,
            sensitivities=sensitivities,
            statistics=statistics,
            fixed_bits=selection_result.fixed_bits,
        )
    
    def _run_baseline(self) -> None:
        """运行FP模式获取baseline输出"""
        log.info("Running baseline forward pass...")
        
        self.model.eval()
        self._baseline_outputs = {}
        
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(
                self.calibration_data,
                desc="Baseline forward",
                leave=False,
            )):
                try:
                    output, prepared_batch = self._run_forward(batch)
                    self._baseline_outputs[batch_idx] = self._prepare_output_for_metric(
                        self._extract_output(output),
                        prepared_batch,
                        move_to_cpu=self.store_baseline_on_cpu,
                    )
                except Exception as e:
                    log.warning(f"Baseline forward failed for batch {batch_idx}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    def _extract_output(self, output: Any) -> torch.Tensor:
        """从模型输出中提取主要张量
        
        Args:
            output: 模型输出
            
        Returns:
            主要输出张量
        """
        if isinstance(output, torch.Tensor):
            return output
        
        if isinstance(output, dict):
            for key in ["logits", "last_hidden_state", "hidden_states"]:
                if key in output:
                    return output[key]
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        
        if isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        
        raise ValueError(f"Cannot extract output tensor from type: {type(output)}")
    
    def _compute_sensitivities(
        self,
        candidates: List[str],
        selection_result: LayerSelectionResult,
    ) -> Dict[str, float]:
        """计算每层敏感度
        
        Args:
            candidates: 候选层列表
            selection_result: 层选择结果
            
        Returns:
            敏感度映射
        """
        log.info("Computing sensitivities...")
        
        sensitivities = {}
        
        for candidate in tqdm(candidates, desc="Computing sensitivities"):
            try:
                sens = self._compute_group_sensitivity(candidate, selection_result)
                sensitivities[candidate] = sens
            except Exception as e:
                log.warning(f"Failed to compute sensitivity for {candidate}: {e}")
                sensitivities[candidate] = 0.0
        
        return sensitivities
    
    def _compute_group_sensitivity(
        self,
        candidate: str,
        selection_result: LayerSelectionResult,
    ) -> float:
        """计算整个组的敏感度（组内所有模块统一量化后计算敏感度）
        
        Args:
            candidate: 候选层标识
            selection_result: 层选择结果
            
        Returns:
            敏感度值
        """
        modules = self.layer_selector.get_modules_for_sensitivity(
            candidate, selection_result
        )
        
        if not modules:
            return 0.0
        
        total_sensitivity = 0.0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(self.calibration_data):
            if batch_idx not in self._baseline_outputs:
                continue
            
            baseline_output = self._baseline_outputs[batch_idx]
            
            try:
                quant_output = self._run_quantized_forward_group(modules, batch)
                if baseline_output.device != quant_output.device:
                    baseline_output = baseline_output.to(quant_output.device)
                sensitivity = self.metric.compute(baseline_output, quant_output)
                total_sensitivity += sensitivity
                valid_batches += 1
            except Exception as e:
                log.debug(f"Sensitivity computation failed for {candidate} batch {batch_idx}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return total_sensitivity / max(valid_batches, 1)
    
    def _run_quantized_forward_group(
        self,
        module_names: List[str],
        batch: Union[List, Dict, torch.Tensor],
    ) -> torch.Tensor:
        """运行量化模式前向传播（组内所有模块统一量化）
        
        Args:
            module_names: 要量化的模块名称列表
            batch: 输入批次
            
        Returns:
            量化后的输出
        """
        original_weights = {}
        
        try:
            for module_name in module_names:
                module = self._get_module(module_name)
                if module is not None and hasattr(module, 'weight'):
                    original_weights[module_name] = module.weight.data.clone()
                    quantized_weight = self._quantize_weight(module.weight.data, self.weight_bits[0])
                    module.weight.data.copy_(quantized_weight)

            with torch.inference_mode():
                output, prepared_batch = self._run_forward(batch)
            return self._prepare_output_for_metric(
                self._extract_output(output),
                prepared_batch,
                move_to_cpu=self.store_baseline_on_cpu,
            )
        finally:
            for module_name, original_weight in original_weights.items():
                module = self._get_module(module_name)
                if module is not None:
                    module.weight.data.copy_(original_weight)
    
    def _compute_module_sensitivity(self, module_name: str) -> float:
        """计算单个模块的敏感度
        
        Args:
            module_name: 模块名称
            
        Returns:
            敏感度值
        """
        total_sensitivity = 0.0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(self.calibration_data):
            if batch_idx not in self._baseline_outputs:
                continue
            
            baseline_output = self._baseline_outputs[batch_idx]
            
            try:
                quant_output = self._run_quantized_forward(module_name, batch)
                if baseline_output.device != quant_output.device:
                    baseline_output = baseline_output.to(quant_output.device)
                sensitivity = self.metric.compute(baseline_output, quant_output)
                total_sensitivity += sensitivity
                valid_batches += 1
            except Exception as e:
                log.debug(f"Sensitivity computation failed for {module_name} batch {batch_idx}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return total_sensitivity / max(valid_batches, 1)
    
    def _run_quantized_forward(
        self,
        module_name: str,
        batch: List,
    ) -> torch.Tensor:
        """运行量化模式前向传播
        
        Args:
            module_name: 要量化的模块名称
            batch: 输入批次
            
        Returns:
            量化后的输出
        """
        module = self._get_module(module_name)
        if module is None:
            raise ValueError(f"Module not found: {module_name}")
        
        original_weight = module.weight.data.clone()
        
        try:
            quantized_weight = self._quantize_weight(module.weight.data, self.weight_bits[0])
            module.weight.data.copy_(quantized_weight)

            with torch.inference_mode():
                output, prepared_batch = self._run_forward(batch)
            return self._prepare_output_for_metric(
                self._extract_output(output),
                prepared_batch,
                move_to_cpu=self.store_baseline_on_cpu,
            )
        finally:
            module.weight.data.copy_(original_weight)
    
    def _get_module(self, module_name: str) -> Optional[nn.Module]:
        """根据名称获取模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            模块实例
        """
        parts = module_name.split(".")
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, "__getitem__"):
                try:
                    module = module[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        
        return module
    
    def _quantize_weight(
        self,
        weight: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        """量化权重（简化版本，用于敏感度计算）
        
        Args:
            weight: 权重张量
            bits: 量化比特数
            
        Returns:
            量化后的权重
        """
        if bits >= 16:
            return weight
        
        origin_dtype = weight.dtype
        weight = weight.float()
        
        min_val = weight.min()
        max_val = weight.max()
        
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = -min_val / scale
        
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2 ** bits - 1)
        
        dequantized = (quantized - zero_point) * scale
        
        return dequantized.to(origin_dtype)

    def _prepare_batch(self, batch: Union[List, Dict, torch.Tensor]) -> Union[List, Dict, torch.Tensor]:
        if isinstance(batch, dict):
            prepared = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            prepared.setdefault("use_cache", False)
            return prepared
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    def _run_forward(self, batch: Union[List, Dict, torch.Tensor]) -> Any:
        prepared = self._prepare_batch(batch)
        if isinstance(prepared, dict):
            return self.model(**prepared), prepared
        if isinstance(prepared, torch.Tensor):
            return self.model(prepared), prepared
        return self.model(*prepared), prepared

    def _prepare_output_for_metric(
        self,
        output: torch.Tensor,
        batch: Union[List, Dict, torch.Tensor],
        move_to_cpu: bool = True,
    ) -> torch.Tensor:
        tensor = output.detach()

        if self.metric_name in {"ppl", "perplexity"}:
            tensor = self._compute_batch_perplexity(tensor, batch)
        else:
            if self.output_reduction == "last_token" and tensor.dim() >= 3:
                tensor = tensor[:, -1, :]
            elif self.output_reduction == "mean_tokens" and tensor.dim() >= 3:
                tensor = tensor.mean(dim=1)

        if move_to_cpu and tensor.device.type != "cpu":
            tensor = tensor.to("cpu")

        return tensor

    def _compute_batch_perplexity(
        self,
        logits: torch.Tensor,
        batch: Union[List, Dict, torch.Tensor],
    ) -> torch.Tensor:
        input_ids, attention_mask = self._extract_labels_from_batch(batch)
        if input_ids is None:
            raise ValueError("PPL metric requires `input_ids` in calibration batch.")
        if logits.dim() < 3:
            raise ValueError("PPL metric requires full logits with shape [batch, seq, vocab].")

        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)

        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].to(loss.device).float()
            valid_tokens = shift_mask.sum().clamp_min(1.0)
            loss = (loss * shift_mask).sum() / valid_tokens
        else:
            loss = loss.mean()

        return torch.exp(loss).reshape(1)

    def _extract_labels_from_batch(
        self,
        batch: Union[List, Dict, torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            return batch.get("input_ids"), batch.get("attention_mask")

        if isinstance(batch, torch.Tensor):
            return batch, None

        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            input_ids = batch[0] if isinstance(batch[0], torch.Tensor) else None
            attention_mask = None
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                attention_mask = batch[1]
            return input_ids, attention_mask

        return None, None

    def _describe_metric(self) -> str:
        if self.metric_name in {"ppl", "perplexity"}:
            return "ppl (quant_ppl - fp_ppl, larger => more sensitive)"

        if self.output_reduction:
            return f"{self.metric_name} [output_reduction={self.output_reduction}] (larger => more sensitive)"

        return f"{self.metric_name} (larger => more sensitive)"
    
    def _allocate_bits(
        self,
        sensitivities: Dict[str, float],
        selection_result: LayerSelectionResult,
    ) -> Dict[str, Dict[str, int]]:
        """根据策略分配比特数
        
        Args:
            sensitivities: 敏感度映射
            selection_result: 层选择结果
            
        Returns:
            比特分配结果
        """
        log.info(f"Allocating bits using policy: {self.policy}")
        
        sorted_candidates = sorted(
            sensitivities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        high_bit = max(self.weight_bits)
        low_bit = min(self.weight_bits)
        
        allocation = {}
        
        if self.policy == "topk":
            topk_count = max(1, int(len(sorted_candidates) * self.topk))
            
            for i, (candidate, _) in enumerate(sorted_candidates):
                bit = high_bit if i < topk_count else low_bit
                allocation[candidate] = {
                    "weight_bits": bit,
                }
        
        elif self.policy == "threshold":
            for candidate, sens in sorted_candidates:
                bit = high_bit if sens > self.threshold else low_bit
                allocation[candidate] = {
                    "weight_bits": bit,
                }
        
        else:
            for candidate, _ in sorted_candidates:
                allocation[candidate] = {
                    "weight_bits": low_bit,
                }
        
        if self.moe_unified:
            allocation = self._unify_moe_bits(allocation, selection_result)
        
        allocation = self._apply_group_unification(allocation, selection_result)
        
        return allocation
    
    def _unify_moe_bits(
        self,
        allocation: Dict[str, Dict[str, int]],
        selection_result: LayerSelectionResult,
    ) -> Dict[str, Dict[str, int]]:
        """统一MoE组内的比特数
        
        Args:
            allocation: 原始分配
            selection_result: 层选择结果
            
        Returns:
            统一后的分配
        """
        unified = {}
        
        for candidate, bits_config in allocation.items():
            if candidate in selection_result.moe_groups:
                members = selection_result.moe_groups[candidate]
                member_bits = [
                    allocation.get(m, bits_config).get("weight_bits", 4)
                    for m in members
                ]
                target_bit = max(member_bits)
                
                unified[candidate] = {
                    "weight_bits": target_bit,
                }
            else:
                unified[candidate] = bits_config
        
        return unified
    
    def _apply_group_unification(
        self,
        allocation: Dict[str, Dict[str, int]],
        selection_result: LayerSelectionResult,
    ) -> Dict[str, Dict[str, int]]:
        """应用自定义分组统一
        
        Args:
            allocation: 原始分配
            selection_result: 层选择结果
            
        Returns:
            统一后的分配
        """
        if not selection_result.group_mapping:
            return allocation
        
        unified = dict(allocation)
        
        group_bits = {}
        for module_name, group_name in selection_result.group_mapping.items():
            if group_name not in group_bits:
                group_bits[group_name] = []
            if module_name in allocation:
                group_bits[group_name].append(allocation[module_name].get("weight_bits", 4))
        
        for group_name, bits_list in group_bits.items():
            if bits_list:
                target_bit = max(bits_list)
                if group_name in unified:
                    unified[group_name] = {"weight_bits": target_bit}
        
        return unified
    
    def _compute_statistics(
        self,
        bit_allocation: Dict[str, Dict[str, int]],
        sensitivities: Dict[str, float],
    ) -> Dict[str, Any]:
        """计算统计信息
        
        Args:
            bit_allocation: 比特分配
            sensitivities: 敏感度
            
        Returns:
            统计信息
        """
        bit_counts = {}
        for config in bit_allocation.values():
            bit = config["weight_bits"]
            bit_counts[bit] = bit_counts.get(bit, 0) + 1
        
        total = len(bit_allocation)
        avg_sensitivity = sum(sensitivities.values()) / max(len(sensitivities), 1)
        
        return {
            "total_layers": total,
            "bit_distribution": bit_counts,
            "avg_sensitivity": avg_sensitivity,
            "high_bit_ratio": bit_counts.get(max(self.weight_bits), 0) / max(total, 1),
        }


def run_mix_precision_search(
    model: nn.Module,
    config: Dict[str, Any],
    calibration_data: List,
    output_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, int]]:
    """执行混合精度搜索的入口函数
    
    这是最小侵入性的集成接口，通过单一函数调用完成所有操作
    
    Args:
        model: 模型实例
        config: 混合精度配置字典
        calibration_data: 校准数据
        output_path: 结果保存路径（可选）
        device: 执行设备
        
    Returns:
        比特分配结果 {layer_name: {"weight_bits": int}}
    """
    engine = MixPrecisionSearchEngine(
        model=model,
        config=config,
        calibration_data=calibration_data,
        device=device,
    )
    
    result = engine.search()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "bit_allocation": result.bit_allocation,
                "sensitivities": result.sensitivities,
                "statistics": result.statistics,
                "fixed_bits": result.fixed_bits,
            }, f, indent=2)
        
        log.info(f"Mix precision search result saved to {output_path}")
    
    return result.bit_allocation


def expand_bit_allocation_to_modules(
    bit_allocation: Dict[str, Dict[str, int]],
    model: nn.Module,
    config: Dict[str, Any],
    fixed_bits: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    """将比特分配展开到所有模块
    
    对于MoE组，将组级别的比特分配展开到组内所有成员
    对于固定bit的层，直接使用固定值
    
    Args:
        bit_allocation: 原始比特分配
        model: 模型实例
        config: 混合精度配置字典
        fixed_bits: 固定bit配置
        
    Returns:
        展开后的比特分配
    """
    layer_selector = LayerSelector(
        include_patterns=config.get("layer_include"),
        exclude_patterns=config.get("layer_exclude"),
        group_by_path=config.get("group_by_path"),
        fixed_bits=config.get("fixed_bits"),
        moe_unified=config.get("moe_unified", True),
    )
    selection_result = layer_selector.select(model)
    
    expanded = {}
    
    for module_name, bits in (fixed_bits or {}).items():
        if bits is not None:
            expanded[module_name] = {"weight_bits": bits}
    
    for candidate, bits_config in bit_allocation.items():
        modules = layer_selector.get_modules_for_sensitivity(candidate, selection_result)
        
        if modules:
            for module_name in modules:
                if module_name not in expanded:
                    expanded[module_name] = bits_config
        else:
            expanded[candidate] = bits_config
    
    return expanded
