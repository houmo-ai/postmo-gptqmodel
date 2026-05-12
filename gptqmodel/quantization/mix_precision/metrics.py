# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class MetricBase(ABC):
    """度量标准基类
    
    所有度量标准必须实现compute方法
    """
    
    @abstractmethod
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        """计算度量值
        
        Args:
            fp_output: 浮点输出
            qt_output: 量化输出
            **kwargs: 额外参数
            
        Returns:
            度量值（敏感度）
        """
        raise NotImplementedError


class KLMetric(MetricBase):
    """KL散度度量
    
    计算FP输出和量化输出之间的KL散度
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        if fp_output.dim() > 2:
            fp_output = fp_output.view(-1, fp_output.size(-1))
            qt_output = qt_output.view(-1, qt_output.size(-1))
        
        fp_log_prob = F.log_softmax(fp_output.float(), dim=-1)
        qt_log_prob = F.log_softmax(qt_output.float(), dim=-1)
        
        kl_div = F.kl_div(
            qt_log_prob,
            fp_log_prob,
            reduction="batchmean",
            log_target=True,
        )
        
        return kl_div.item()


class SQNRMetric(MetricBase):
    """信号量化噪声比度量
    
    计算噪声与信号的比值
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        noise = fp_output.float() - qt_output.float()
        signal_power = (fp_output.float() ** 2).mean() + 1e-10
        noise_power = (noise ** 2).mean()
        
        return (noise_power / signal_power).item()


class L1Metric(MetricBase):
    """L1距离度量
    
    计算FP输出和量化输出之间的L1距离
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        diff = (fp_output.float() - qt_output.float()).abs()
        return (diff.sum() / diff.numel()).item()


class MSEMetric(MetricBase):
    """均方误差度量
    
    计算FP输出和量化输出之间的MSE
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        diff = fp_output.float() - qt_output.float()
        return (diff ** 2).mean().item()


class HessianMSEMetric(MetricBase):
    """Hessian加权MSE度量
    
    使用Hessian矩阵加权计算MSE
    这是xhquant中使用的主要度量标准
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> float:
        diff = fp_output.float() - qt_output.float()
        
        if hessian is not None:
            if hessian.dim() == 2:
                weighted_diff = torch.matmul(diff, hessian)
                mse = (weighted_diff * diff).sum() / diff.numel()
            else:
                mse = (diff ** 2).mean()
        else:
            mse = (diff ** 2).mean()
        
        return mse.item()


class CosineSimilarityMetric(MetricBase):
    """余弦相似度度量
    
    计算FP输出和量化输出之间的余弦相似度
    值越小表示敏感度越高（相似度越低）
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        fp_flat = fp_output.float().flatten()
        qt_flat = qt_output.float().flatten()
        
        cos_sim = F.cosine_similarity(
            fp_flat.unsqueeze(0),
            qt_flat.unsqueeze(0),
        )
        
        return 1.0 - cos_sim.item()


class MaxAbsDiffMetric(MetricBase):
    """最大绝对差度量
    
    计算FP输出和量化输出之间的最大绝对差
    值越大表示敏感度越高
    """
    
    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        diff = (fp_output.float() - qt_output.float()).abs()
        return diff.max().item()


class PerplexityMetric(MetricBase):
    """Perplexity delta metric.

    Expects ``fp_output`` and ``qt_output`` to already be per-batch perplexity
    scalars. The sensitivity is defined as ``quant_ppl - fp_ppl`` so larger
    values indicate larger degradation after quantization.
    """

    def compute(
        self,
        fp_output: torch.Tensor,
        qt_output: torch.Tensor,
        **kwargs,
    ) -> float:
        return (qt_output.float().mean() - fp_output.float().mean()).item()


class MetricFactory:
    """度量标准工厂
    
    根据名称创建对应的度量标准实例
    """
    
    _metrics = {
        "kl": KLMetric,
        "kld": KLMetric,
        "sqnr": SQNRMetric,
        "l1": L1Metric,
        "mse": MSEMetric,
        "hessian_mse": HessianMSEMetric,
        "hessian-mse": HessianMSEMetric,
        "hmse": HessianMSEMetric,
        "cosine": CosineSimilarityMetric,
        "cos_sim": CosineSimilarityMetric,
        "max_abs": MaxAbsDiffMetric,
        "ppl": PerplexityMetric,
        "perplexity": PerplexityMetric,
    }
    
    @classmethod
    def create(cls, metric_name: str) -> MetricBase:
        """创建度量标准实例
        
        Args:
            metric_name: 度量标准名称
            
        Returns:
            度量标准实例
            
        Raises:
            ValueError: 不支持的度量标准名称
        """
        metric_name_lower = metric_name.lower()
        
        if metric_name_lower not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available metrics: {available}"
            )
        
        return cls._metrics[metric_name_lower]()
    
    @classmethod
    def register(cls, name: str, metric_class: type) -> None:
        """注册自定义度量标准
        
        Args:
            name: 度量标准名称
            metric_class: 度量标准类
        """
        cls._metrics[name.lower()] = metric_class
    
    @classmethod
    def list_available(cls) -> list:
        """列出所有可用的度量标准
        
        Returns:
            可用度量标准名称列表
        """
        return list(cls._metrics.keys())


def compute_sensitivity(
    fp_output: torch.Tensor,
    qt_output: torch.Tensor,
    metric: str = "kl",
    **kwargs,
) -> float:
    """计算敏感度的便捷函数
    
    Args:
        fp_output: 浮点输出
        qt_output: 量化输出
        metric: 度量标准名称
        **kwargs: 传递给度量标准的额外参数
        
    Returns:
        敏感度值
    """
    metric_instance = MetricFactory.create(metric)
    return metric_instance.compute(fp_output, qt_output, **kwargs)
