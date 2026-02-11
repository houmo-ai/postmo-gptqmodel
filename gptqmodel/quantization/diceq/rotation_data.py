"""Data loading and sampling helpers for DiceQ rotation training."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TensorFileMeta:
    """Metadata for a single tensor file in the training folder."""

    path: Path
    layer_idx: int
    sample_idx: Optional[int]
    format_type: str  # sample_separate | layer_combined | unknown


@dataclass(frozen=True)
class SampleRef:
    """Reference to one logical training sample."""

    path: Path
    layer_idx: int
    inner_idx: Optional[int] = None
    score: Optional[float] = None


class ActivationReferenceDataset(Dataset):
    """Lazy dataset that loads tensors by reference with lightweight caching."""

    def __init__(self, refs: List[SampleRef]):
        self.refs = refs
        self._cached_path: Optional[Path] = None
        self._cached_tensor: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ref = self.refs[idx]
        tensor = self._load_tensor(ref.path)
        sample = tensor if ref.inner_idx is None else tensor[ref.inner_idx]

        if sample.ndim == 3 and sample.shape[0] == 1:
            sample = sample.squeeze(0)
        if sample.ndim != 2:
            raise ValueError(
                f"Sample must be 2D after normalization, got {tuple(sample.shape)} "
                f"from file {ref.path}"
            )
        return sample

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if self._cached_path == path and self._cached_tensor is not None:
            return self._cached_tensor
        tensor = safe_torch_load(path)
        if not torch.is_tensor(tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
        self._cached_path = path
        self._cached_tensor = tensor
        return tensor


class WeightReferenceDataset(Dataset):
    """Lazy dataset for loading weight tensors by reference with lightweight caching.
    
    Similar to ActivationReferenceDataset but designed for weight data.
    Weight tensors are expected to be 2D with shape [oc, ic] (output_channels, input_channels).
    """

    def __init__(self, refs: List[SampleRef]):
        self.refs = refs
        self._cached_path: Optional[Path] = None
        self._cached_tensor: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ref = self.refs[idx]
        tensor = self._load_tensor(ref.path)
        sample = tensor if ref.inner_idx is None else tensor[ref.inner_idx]

        # Weight tensors should be 2D [oc, ic]
        if sample.ndim == 3 and sample.shape[0] == 1:
            sample = sample.squeeze(0)
        if sample.ndim != 2:
            raise ValueError(
                f"Weight sample must be 2D [oc, ic], got {tuple(sample.shape)} "
                f"from file {ref.path}"
            )
        return sample

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if self._cached_path == path and self._cached_tensor is not None:
            return self._cached_tensor
        tensor = safe_torch_load(path)
        if not torch.is_tensor(tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
        self._cached_path = path
        self._cached_tensor = tensor
        return tensor


def safe_torch_load(path: Path) -> torch.Tensor:
    """Load tensor from disk in a PyTorch-version compatible way."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _compute_sample_score(sample: torch.Tensor, score_metric: str) -> float:
    sample_abs = sample.float().abs().reshape(-1)
    if score_metric == "absmax":
        return float(sample_abs.max().item())
    if score_metric == "p99":
        return float(torch.quantile(sample_abs, torch.tensor(0.99)).item())
    if score_metric == "p99_over_median":
        p99 = float(torch.quantile(sample_abs, torch.tensor(0.99)).item())
        med = float(torch.quantile(sample_abs, torch.tensor(0.5)).item())
        return p99 / max(med, 1e-6)
    raise ValueError(f"Unsupported score metric: {score_metric}")


def resolve_data_dir(
    args: argparse.Namespace,
    target_name: str,
    entry: Dict[str, Any],
) -> Path:
    """Resolve concrete training-data directory from CLI or config."""
    if args.data_path:
        data_dir = Path(args.data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"--data_path does not exist: {data_dir}")
        return data_dir

    folder = str(entry.get("folder", target_name))
    if not folder:
        raise ValueError(
            f"Config entry '{target_name}' must define non-empty 'folder'."
        )
    if not args.model:
        raise ValueError("--model is required when --data_path is not provided.")

    model_name = args.model.split("/")[-1]
    base = Path(args.data_root) / f"{args.calib_dataset}_{args.calib_sample}samples" / model_name
    candidates = [base / folder, base / f"{folder}_train"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No data directory found for target '{target_name}'. Checked: {candidates}"
    )


def scan_tensor_files(data_dir: Path, nsamples: int) -> List[TensorFileMeta]:
    """Scan all .pt files and parse metadata from filenames."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    pt_files = sorted([p for p in data_dir.iterdir() if p.suffix == ".pt"])
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}.")

    sample_pattern = re.compile(r"sample_(\d+)_layer_(\d+)_.*\.pt$")
    layer_pattern = re.compile(r"layer_(\d+)_.*\.pt$")
    metas: List[TensorFileMeta] = []
    for path in pt_files:
        name = path.name
        sample_match = sample_pattern.match(name)
        if sample_match:
            sample_idx = int(sample_match.group(1))
            if sample_idx >= nsamples:
                continue
            layer_idx = int(sample_match.group(2))
            metas.append(
                TensorFileMeta(
                    path=path,
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                    format_type="sample_separate",
                )
            )
            continue

        layer_match = layer_pattern.match(name)
        if layer_match:
            metas.append(
                TensorFileMeta(
                    path=path,
                    layer_idx=int(layer_match.group(1)),
                    sample_idx=None,
                    format_type="layer_combined",
                )
            )
            continue

        metas.append(
            TensorFileMeta(
                path=path,
                layer_idx=0,
                sample_idx=None,
                format_type="unknown",
            )
        )

    if not metas:
        raise ValueError(
            f"No usable files after nsamples filtering (nsamples={nsamples}) in {data_dir}."
        )
    return metas


def build_sample_refs(
    metas: List[TensorFileMeta],
    nsamples: int,
    seed: int = 0,
    score_metric: Optional[str] = None,
) -> Dict[int, List[SampleRef]]:
    """Convert file metadata to per-layer sample references."""
    if any(meta.format_type == "sample_separate" for meta in metas):
        sample_only = [meta for meta in metas if meta.format_type == "sample_separate"]
        if len(sample_only) != len(metas):
            metas = sample_only

    rng = np.random.default_rng(seed)
    refs_by_layer: Dict[int, List[SampleRef]] = {}
    for meta in metas:
        refs_by_layer.setdefault(meta.layer_idx, [])
        if meta.format_type == "sample_separate":
            score = None
            if score_metric is not None:
                sample_tensor = safe_torch_load(meta.path)
                if not torch.is_tensor(sample_tensor):
                    raise ValueError(f"Expected tensor in {meta.path}, got {type(sample_tensor)}")
                score = _compute_sample_score(sample_tensor, score_metric=score_metric)
            refs_by_layer[meta.layer_idx].append(
                SampleRef(path=meta.path, layer_idx=meta.layer_idx, inner_idx=None, score=score)
            )
            continue

        tensor = safe_torch_load(meta.path)
        if not torch.is_tensor(tensor):
            raise ValueError(f"Expected tensor in {meta.path}, got {type(tensor)}")
        if tensor.ndim == 2:
            score = _compute_sample_score(tensor, score_metric=score_metric) if score_metric is not None else None
            refs_by_layer[meta.layer_idx].append(
                SampleRef(path=meta.path, layer_idx=meta.layer_idx, inner_idx=None, score=score)
            )
            continue
        if tensor.ndim != 3:
            raise ValueError(
                f"Unsupported tensor rank {tensor.ndim} in {meta.path}; expected 2D or 3D."
            )
        ref_count = min(tensor.shape[0], nsamples)
        if ref_count == tensor.shape[0]:
            selected_indices = np.arange(ref_count, dtype=np.int64)
        else:
            # Random subset sampling without replacement for better coverage.
            selected_indices = rng.choice(tensor.shape[0], size=ref_count, replace=False)
            selected_indices.sort()
        for idx in selected_indices.tolist():
            score = None
            if score_metric is not None:
                score = _compute_sample_score(tensor[idx], score_metric=score_metric)
            refs_by_layer[meta.layer_idx].append(
                SampleRef(
                    path=meta.path,
                    layer_idx=meta.layer_idx,
                    inner_idx=int(idx),
                    score=score,
                )
            )
    return refs_by_layer
