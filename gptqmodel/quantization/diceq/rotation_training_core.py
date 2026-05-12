"""Core training logic for DiceQ rotation matrix optimization."""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None

try:
    import geoopt
except ImportError:  # pragma: no cover
    geoopt = None

from .rotation_data import ActivationReferenceDataset, SampleRef, WeightReferenceDataset
from ..rotation.hadamard_utils import random_hadamard_matrix, random_block_hadamard_matrix


def random_orthogonal_matrix(size: int, device) -> torch.Tensor:
    """Generate a random orthogonal matrix using QR decomposition."""
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size: int, mode: str, device) -> torch.Tensor:
    """Get orthogonal matrix based on specified mode."""
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    elif mode == "block_hadamard":
        # Use default block size; falls back to full hadamard if not compatible
        return random_block_hadamard_matrix(size, device)
    elif mode == "identity":
        return torch.eye(size, dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unknown mode {mode}")


class R_QR(nn.Module):
    """Global/layer-wise orthogonal rotation module (same core as r1_base_qr.py).
    
    This module implements A@R transformation for activation training:
    - A: activation tensor [batch*seq_len, hidden_size] or [seq_len, hidden_size]
    - R: rotation matrix [hidden_size, hidden_size]
    - Output: A@R
    
    Consistency Note:
    - Training: A@R (activation right-multiplied by rotation matrix)
    - Application: Input layers use W@R, Output layers use R.T@W
    - Mathematical equivalence: (A@R)@W = A@(R@W), but application uses A@(W@R)
    """

    def __init__(self, hidden_size: int, use_geoopt: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_geoopt = use_geoopt
        if use_geoopt:
            if geoopt is None:
                raise ImportError(
                    "geoopt is required for optimizer RSGD/RAdam, "
                    "but it is not installed in the current environment."
                )
            self.matrix = geoopt.ManifoldParameter(
                torch.eye(hidden_size),
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix = nn.Parameter(torch.eye(hidden_size))
        self.rotate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_geoopt:
            self.rotate = self.matrix
        else:
            self.rotate, _ = torch.linalg.qr(self.matrix, mode="complete")
        return torch.matmul(x, self.rotate)


class HeadWiseQR(nn.Module):
    """Per-head orthogonal rotation module (aligned with r2_base_qr.py style)."""

    def __init__(self, hidden_size: int, head_num: int, use_geoopt: bool = False):
        super().__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by head_num ({head_num})."
            )
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = hidden_size // head_num
        self.use_geoopt = use_geoopt

        init = torch.eye(self.head_dim).repeat(self.head_num, 1, 1)
        if use_geoopt:
            if geoopt is None:
                raise ImportError(
                    "geoopt is required for optimizer RSGD/RAdam, "
                    "but it is not installed in the current environment."
                )
            self.matrix = geoopt.ManifoldParameter(
                init,
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix = nn.Parameter(init)
        self.rotate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.reshape(-1, self.head_num, self.head_dim).transpose(0, 1)
        if self.use_geoopt:
            self.rotate = self.matrix
        else:
            self.rotate, _ = torch.linalg.qr(self.matrix)
        x = torch.matmul(x, self.rotate)
        return x.transpose(0, 1).reshape(x_shape)


class R_QR_Weight(nn.Module):
    """Weight-based orthogonal rotation module for training rotation matrices from weights.
    
    This module implements W@R transformation where:
    - W: weight tensor [oc, ic] (output_channels, input_channels)
    - R: rotation matrix [ic, ic]
    - Output: W@R [oc, ic]
    
    Consistency Note:
    - Training: W@R (matches input channel layer application: W@R)
    - Application: Input layers use W@R, Output layers use R.T@W
    - For output layers trained with weights, ensure consistency with R.T@W application
    """

    def __init__(self, input_size: int, use_geoopt: bool = False):
        super().__init__()
        self.input_size = input_size
        self.use_geoopt = use_geoopt
        if use_geoopt:
            if geoopt is None:
                raise ImportError(
                    "geoopt is required for optimizer RSGD/RAdam, "
                    "but it is not installed in the current environment."
                )
            self.matrix = geoopt.ManifoldParameter(
                torch.eye(input_size),
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix = nn.Parameter(torch.eye(input_size))
        self.rotate = None

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to weight tensor.
        
        Args:
            W: Weight tensor of shape [batch, oc, ic] or [oc, ic]
            
        Returns:
            Rotated weight tensor W@R with same shape as input
        """
        if self.use_geoopt:
            self.rotate = self.matrix
        else:
            self.rotate, _ = torch.linalg.qr(self.matrix, mode="complete")
        # W@R: [oc, ic] @ [ic, ic] -> [oc, ic]
        return torch.matmul(W, self.rotate)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_hidden_size_from_dataset(dataset: Dataset, data_type: str = "activation") -> int:
    """Infer hidden size from the first sample.
    
    Args:
        dataset: Dataset to infer from
        data_type: "activation" or "weight"
            - activation: sample shape is [seq_len, hidden_size], infer from last dim
            - weight: sample shape is [oc, ic], infer from last dim (ic)
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer hidden size.")
    sample = dataset[0]
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D sample, got shape {tuple(sample.shape)}.")
    hidden_size = int(sample.shape[-1])
    if hidden_size <= 0:
        raise ValueError(f"Invalid hidden size inferred: {hidden_size}")
    return hidden_size


def infer_head_num(
    args: argparse.Namespace,
    train_args: Dict[str, Any],
    hidden_size: int,
) -> int:
    """Infer head_num for head-wise mode from config, CLI, or model config."""
    if train_args.get("head_num") is not None:
        head_num = int(train_args["head_num"])
    elif args.head_num is not None:
        head_num = int(args.head_num)
    else:
        if not args.model:
            raise ValueError(
                "head_wise mode requires head_num, but --model is not provided."
            )
        if transformers is None:
            raise ImportError(
                "transformers is required to infer head_num from model config."
            )
        config = transformers.AutoConfig.from_pretrained(
            args.model,
            token=args.hf_token,
            trust_remote_code=True,
        )
        if not hasattr(config, "num_attention_heads"):
            raise ValueError(
                "Cannot infer head_num from model config: missing num_attention_heads."
            )
        head_num = int(config.num_attention_heads)

    if hidden_size % head_num != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by head_num ({head_num})."
        )
    return head_num


def build_optimizer(
    module: nn.Module,
    train_args: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Create optimizer based on validated train args."""
    lr = float(train_args["learning_rate"])
    momentum = float(train_args["momentum"])
    weight_decay = float(train_args["weight_decay"])
    optimizer_name = train_args["optimizer"]

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            module.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "RSGD":
        if geoopt is None:
            raise ImportError("geoopt is not installed; cannot use RSGD.")
        return geoopt.optim.RiemannianSGD(
            module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            stabilize=10,
        )
    if optimizer_name == "RAdam":
        if geoopt is None:
            raise ImportError("geoopt is not installed; cannot use RAdam.")
        return geoopt.optim.RiemannianAdam(
            module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            stabilize=10,
        )
    raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")


def compute_qsr_loss(outputs: torch.Tensor, train_args: Dict[str, Any]) -> torch.Tensor:
    """Q-SR loss: keep same matrix-loss core logic as r1_base_qr.py."""
    q_max = float(train_args["qsr_qmax"])
    alpha = float(train_args["qsr_alpha"])
    tau = float(train_args["qsr_tau"])
    eps = float(train_args["qsr_eps"])
    loss_scale = float(train_args["loss_scale"])

    d = outputs.shape[-1]
    smooth_abs = (outputs.pow(2) + eps ** 2).sqrt()
    rho = outputs.norm(p=2, dim=-1, keepdim=True) + eps
    s = smooth_abs / rho

    log_d = math.log(d)
    m_alpha = (torch.logsumexp(alpha * s, dim=-1, keepdim=True) - log_d) / alpha
    tilde_a = rho * m_alpha

    t = m_alpha / (2 * q_max)
    u_gate = torch.sigmoid((t - s) / tau)
    tilde_u = u_gate.mean(dim=-1, keepdim=True)

    scale_term = tilde_a.pow(2)
    distortion_term = (1.0 / 12.0) + 0.25 * tilde_u
    return (scale_term * distortion_term).mean() * loss_scale


def create_train_dataloader(
    dataset: Dataset,
    train_args: Dict[str, Any],
    sampler: Optional[SubsetRandomSampler] = None,
) -> DataLoader:
    """Build dataloader with mode-safe worker settings."""
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(train_args["train_batch_size"]),
        "sampler": sampler,
        "shuffle": sampler is None,
    }
    num_workers = int(train_args["num_workers"])
    pin_memory = bool(train_args["pin_memory"]) and torch.cuda.is_available()
    kwargs["pin_memory"] = pin_memory

    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["prefetch_factor"] = int(train_args["prefetch_factor"])
        kwargs["persistent_workers"] = True
    else:
        kwargs["num_workers"] = 0
    return DataLoader(**kwargs)


@torch.no_grad()
def evaluate_rotation_loss(
    model: nn.Module,
    dataset: Dataset,
    hidden_size: int,
    train_args: Dict[str, Any],
    device: torch.device,
    max_batches: int = 20,
    data_type: str = "activation",
) -> float:
    """Evaluate trained rotation matrix on held-out pass over training data."""
    model.eval()
    dataloader = create_train_dataloader(dataset, train_args, sampler=None)
    losses: List[float] = []
    for batch_idx, batch_samples in enumerate(dataloader):
        batch_samples = batch_samples.to(device).float()
        
        if data_type == "weight":
            # Weight data: keep shape as-is; torch.matmul supports batched matmul.
            outputs = model(batch_samples)
        else:
            # Activation data: reshape to [batch*seq_len, hidden_size]
            batch_samples = batch_samples.reshape(-1, hidden_size)
            outputs = model(batch_samples)
        
        losses.append(float(compute_qsr_loss(outputs, train_args).item()))
        if max_batches > 0 and batch_idx + 1 >= max_batches:
            break
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def get_effective_rotation(model: nn.Module) -> torch.Tensor:
    """Get orthogonal matrix used in forward pass."""
    if getattr(model, "use_geoopt", False):
        rotate = model.matrix
    else:
        rotate, _ = torch.linalg.qr(model.matrix)
    return rotate.detach().cpu()


def compute_orthogonality_error(rotation: torch.Tensor) -> float:
    """Compute normalized orthogonality error for a rotation matrix."""
    if rotation.ndim == 2:
        identity = torch.eye(rotation.shape[0], dtype=rotation.dtype)
        diff = rotation.T @ rotation - identity
        return float(torch.linalg.norm(diff, ord="fro").item() / rotation.shape[0])
    if rotation.ndim == 3:
        errors: List[float] = []
        for i in range(rotation.shape[0]):
            head = rotation[i]
            identity = torch.eye(head.shape[0], dtype=head.dtype)
            diff = head.T @ head - identity
            errors.append(float(torch.linalg.norm(diff, ord="fro").item() / head.shape[0]))
        return float(np.mean(errors)) if errors else float("nan")
    return float("nan")


def _collect_dataset_scores(dataset: Dataset) -> Optional[np.ndarray]:
    """Collect precomputed sample scores from dataset refs (if available)."""
    refs = getattr(dataset, "refs", None)
    if not isinstance(refs, list) or len(refs) != len(dataset):
        return None

    scores: List[float] = []
    for ref in refs:
        score = getattr(ref, "score", None)
        if score is None:
            return None
        score_value = float(score)
        if not np.isfinite(score_value):
            return None
        scores.append(score_value)
    return np.asarray(scores, dtype=np.float64)


def _sample_epoch_indices(
    *,
    dataset_size: int,
    subset_size: int,
    train_args: Dict[str, Any],
    score_array: Optional[np.ndarray],
) -> np.ndarray:
    """Sample per-epoch subset indices according to sampling policy."""
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be > 0, got {dataset_size}.")
    subset_size = max(1, min(int(subset_size), int(dataset_size)))

    sampling_policy = str(train_args.get("sampling_policy", "random"))
    if sampling_policy != "outlier_mix" or score_array is None or score_array.shape[0] != dataset_size:
        return np.random.choice(dataset_size, size=subset_size, replace=False)

    outlier_mix_ratio = float(train_args.get("outlier_mix_ratio", 0.0))
    if outlier_mix_ratio <= 0.0:
        return np.random.choice(dataset_size, size=subset_size, replace=False)

    outlier_topk_ratio = float(train_args.get("outlier_topk_ratio", 0.2))
    outlier_take = int(round(subset_size * outlier_mix_ratio))
    outlier_take = max(0, min(outlier_take, subset_size))
    if outlier_take == 0:
        return np.random.choice(dataset_size, size=subset_size, replace=False)

    pool_size = max(1, min(dataset_size, int(math.ceil(dataset_size * outlier_topk_ratio))))
    topk_pool = np.argpartition(score_array, -pool_size)[-pool_size:]
    outlier_take = min(outlier_take, int(topk_pool.shape[0]))

    selected_chunks: List[np.ndarray] = []
    selected_mask = np.zeros(dataset_size, dtype=bool)
    if outlier_take > 0:
        outlier_idx = np.random.choice(topk_pool, size=outlier_take, replace=False).astype(np.int64, copy=False)
        selected_chunks.append(outlier_idx)
        selected_mask[outlier_idx] = True

    remaining_take = subset_size - outlier_take
    if remaining_take > 0:
        remaining_pool = np.flatnonzero(~selected_mask)
        random_idx = np.random.choice(remaining_pool, size=remaining_take, replace=False).astype(np.int64, copy=False)
        selected_chunks.append(random_idx)

    indices = np.concatenate(selected_chunks, axis=0) if selected_chunks else np.empty((0,), dtype=np.int64)
    np.random.shuffle(indices)
    return indices


def train_one_rotation(
    dataset: Dataset,
    hidden_size: int,
    train_args: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    train_name: str,
    head_num: Optional[int] = None,
    wandb_run: Any = None,
    show_progress: bool = True,
    data_type: str = "activation",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train one rotation module and return trained matrix + metrics.
    
    Args:
        dataset: Dataset containing training samples
        hidden_size: Hidden size (for activation) or input channel size (for weight)
        train_args: Training arguments dictionary
        device: Device to train on
        logger: Logger instance
        train_name: Name for logging
        head_num: Optional head number for head-wise training
        wandb_run: Optional wandb run
        show_progress: Whether to show progress bars
        data_type: "activation" or "weight" - determines training mode
    """
    use_geoopt = train_args["optimizer"] in {"RSGD", "RAdam"}
    effective_train_args: Dict[str, Any] = dict(train_args)
    
    if data_type == "weight":
        # Weight training: use R_QR_Weight which applies W@R
        if head_num is not None:
            raise ValueError("head_num is not supported for weight training mode")
        model = R_QR_Weight(input_size=hidden_size, use_geoopt=use_geoopt).to(device)
        init_r = get_orthogonal_matrix(
            hidden_size,
            train_args["init_mode"],
            device=device,
        ).float()
        model.matrix.data.copy_(init_r)
        # Weight samples are 2D matrices [oc, ic]. Different modules can have different oc,
        # which breaks DataLoader stacking when batch_size > 1.
        # To keep behavior robust with minimal config changes, force batch_size=1 in weight mode.
        if int(effective_train_args.get("train_batch_size", 1)) != 1:
            logger.warning(
                "[%s] weight training forces train_batch_size=1 (was %s) to avoid stacking variable [oc, ic] tensors.",
                train_name,
                effective_train_args.get("train_batch_size"),
            )
            effective_train_args["train_batch_size"] = 1
        # Also avoid spawning workers for tiny batch=1 weight matrices.
        effective_train_args["num_workers"] = 0
        effective_train_args["prefetch_factor"] = None
        effective_train_args["pin_memory"] = False
    else:
        # Activation training: use R_QR or HeadWiseQR
        if head_num is None:
            model = R_QR(hidden_size=hidden_size, use_geoopt=use_geoopt).to(device)
            init_r = get_orthogonal_matrix(
                hidden_size,
                train_args["init_mode"],
                device=device,
            ).float()
        else:
            model = HeadWiseQR(
                hidden_size=hidden_size,
                head_num=head_num,
                use_geoopt=use_geoopt,
            ).to(device)
            head_dim = hidden_size // head_num
            init_head = get_orthogonal_matrix(
                head_dim,
                train_args["init_mode"],
                device=device,
            ).float()
            init_r = init_head.unsqueeze(0).repeat(head_num, 1, 1)
        model.matrix.data.copy_(init_r)

    optimizer = build_optimizer(model, effective_train_args)
    scheduler = None
    if effective_train_args["cos_lr"]:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(effective_train_args["num_epochs"]),
            eta_min=0.0,
        )

    num_epochs = int(effective_train_args["num_epochs"])
    accumulation_steps = int(effective_train_args["accumulation_steps"])
    subset_size_cfg = effective_train_args.get("subset_size")
    if subset_size_cfg is not None:
        if isinstance(subset_size_cfg, float):
            # subset_size is a ratio (like train_subset_size)
            subset_size = max(1, int(len(dataset) * subset_size_cfg))
        else:
            # subset_size is an integer count
            subset_size = max(1, min(int(subset_size_cfg), len(dataset)))
    else:
        subset_ratio = float(effective_train_args["train_subset_size"])
        subset_size = max(1, int(len(dataset) * subset_ratio))
    sampling_policy = str(effective_train_args.get("sampling_policy", "random"))
    score_array = _collect_dataset_scores(dataset) if sampling_policy == "outlier_mix" else None
    if sampling_policy == "outlier_mix" and score_array is None:
        logger.warning(
            "[%s] sampling_policy=outlier_mix but score is unavailable; fallback to random sampling.",
            train_name,
        )
    if sampling_policy == "outlier_mix":
        logger.info(
            "[%s] outlier-mix enabled: mix_ratio=%.3f, topk_ratio=%.3f",
            train_name,
            float(train_args.get("outlier_mix_ratio", 0.0)),
            float(train_args.get("outlier_topk_ratio", 0.2)),
        )

    logger.info(
        "[%s] start training: samples=%d, subset_size=%d, epochs=%d, bsz=%d, sampling_policy=%s",
        train_name,
        len(dataset),
        subset_size,
        num_epochs,
        int(effective_train_args["train_batch_size"]),
        sampling_policy,
    )

    epoch_losses: List[float] = []
    for epoch in tqdm(
        range(num_epochs),
        desc=f"Epochs<{train_name}>",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = _sample_epoch_indices(
            dataset_size=len(dataset),
            subset_size=subset_size,
            train_args=effective_train_args,
            score_array=score_array,
        )
        sampler = SubsetRandomSampler(indices.tolist())
        dataloader = create_train_dataloader(dataset, effective_train_args, sampler=sampler)

        loss_log: List[torch.Tensor] = []
        pending_batches = 0
        batch_bar = tqdm(
            dataloader,
            desc=f"Train<{train_name}> e{epoch + 1}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for batch_idx, batch_samples in enumerate(batch_bar):
            batch_samples = batch_samples.to(device).float()
            
            if data_type == "weight":
                # Weight data: keep shape as-is; torch.matmul supports batched matmul.
                # batch_samples is typically [1, oc, ic] when train_batch_size=1.
                outputs = model(batch_samples)
            else:
                # Activation data: reshape to [batch*seq_len, hidden_size]
                batch_samples = batch_samples.reshape(-1, hidden_size)
                outputs = model(batch_samples)
            
            loss = compute_qsr_loss(outputs, effective_train_args)
            loss = loss / accumulation_steps
            loss.backward()
            pending_batches += 1

            if pending_batches == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_batches = 0

            loss_log.append(loss.detach())
            batch_bar.set_postfix(loss=float(loss.item()))

        # Handle leftover gradients when number of batches is not divisible by accumulation_steps.
        if pending_batches > 0:
            scale = float(accumulation_steps) / float(pending_batches)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        mean_loss = float(torch.stack(loss_log).mean().item()) if loss_log else float("nan")
        epoch_losses.append(mean_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "[%s] epoch %d/%d, loss=%.8f, lr=%.6e",
            train_name,
            epoch + 1,
            num_epochs,
            mean_loss,
            lr_now,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{train_name}/epoch": epoch + 1,
                    f"{train_name}/train_loss": mean_loss,
                    f"{train_name}/lr": lr_now,
                }
            )

    skip_eval = bool(effective_train_args.get("skip_eval", False))
    if skip_eval:
        eval_loss = float("nan")
        logger.info("[%s] evaluation skipped (skip_eval=True)", train_name)
    else:
        eval_max_batches = int(effective_train_args.get("eval_max_batches", 4))
        eval_loss = evaluate_rotation_loss(
            model=model,
            dataset=dataset,
            hidden_size=hidden_size,
            train_args=effective_train_args,
            device=device,
            max_batches=eval_max_batches,
            data_type=data_type,
        )
        logger.info("[%s] evaluation loss=%.8f (max_batches=%d)", train_name, eval_loss, eval_max_batches)
        if wandb_run is not None:
            wandb_run.log({f"{train_name}/eval_loss": eval_loss})

    rotation = get_effective_rotation(model)
    orthogonality_error = compute_orthogonality_error(rotation)
    logger.info("[%s] orthogonality_error=%.6e", train_name, orthogonality_error)
    metrics = {
        "final_train_loss": epoch_losses[-1] if epoch_losses else float("nan"),
        "eval_loss": eval_loss,
        "orthogonality_error": orthogonality_error,
    }
    return rotation, metrics


def _build_worker_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"r1_config_qr.worker.{os.getpid()}.{name}")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _train_single_layer_task(
    *,
    layer_idx: int,
    refs: List[SampleRef],
    target_name: str,
    hidden_size: int,
    train_args: Dict[str, Any],
    device_arg: str,
    head_num: Optional[int],
    seed: int,
    data_type: str = "activation",
) -> Tuple[int, torch.Tensor, Dict[str, float]]:
    set_random_seed(int(seed) + int(layer_idx))
    if data_type == "weight":
        dataset = WeightReferenceDataset(refs)
    else:
        dataset = ActivationReferenceDataset(refs)
    worker_logger = _build_worker_logger(f"{target_name}_layer_{layer_idx}")
    device_obj = resolve_device(device_arg)
    rotation, metrics = train_one_rotation(
        dataset=dataset,
        hidden_size=hidden_size,
        train_args=train_args,
        device=device_obj,
        logger=worker_logger,
        train_name=f"{target_name}_layer_{layer_idx}",
        head_num=head_num,
        wandb_run=None,
        show_progress=False,
        data_type=data_type,
    )
    return int(layer_idx), rotation, metrics


def train_all_mode(
    refs_by_layer: Dict[int, List[SampleRef]],
    target_name: str,
    train_args: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    wandb_run: Any = None,
    data_type: str = "activation",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]], int]:
    """Train one shared rotation matrix over all layers' activations or weights."""
    all_refs: List[SampleRef] = []
    for layer_idx in sorted(refs_by_layer):
        all_refs.extend(refs_by_layer[layer_idx])
    
    if data_type == "weight":
        dataset = WeightReferenceDataset(all_refs)
    else:
        dataset = ActivationReferenceDataset(all_refs)
    
    hidden_size = infer_hidden_size_from_dataset(dataset, data_type=data_type)
    rotation, metrics = train_one_rotation(
        dataset=dataset,
        hidden_size=hidden_size,
        train_args=train_args,
        device=device,
        logger=logger,
        train_name=target_name,
        head_num=None,
        wandb_run=wandb_run,
        data_type=data_type,
    )
    return {target_name: rotation}, {target_name: metrics}, hidden_size


def train_layer_or_head_wise_mode(
    refs_by_layer: Dict[int, List[SampleRef]],
    target_name: str,
    train_args: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace,
    wandb_run: Any = None,
    data_type: str = "activation",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], int]:
    """Train per-layer matrices in layer-wise or head-wise mode."""
    train_mode = train_args["train_data"]
    is_head_wise = train_mode == "head_wise"
    rotation_dict: Dict[str, Any] = {}
    metrics_dict: Dict[str, Dict[str, float]] = {}

    sorted_layers = sorted(refs_by_layer)
    if not sorted_layers:
        raise ValueError("No layers found for layer/head-wise training.")

    if data_type == "weight":
        first_dataset = WeightReferenceDataset(refs_by_layer[sorted_layers[0]])
    else:
        first_dataset = ActivationReferenceDataset(refs_by_layer[sorted_layers[0]])
    hidden_size = infer_hidden_size_from_dataset(first_dataset, data_type=data_type)
    head_num = infer_head_num(args, train_args, hidden_size) if is_head_wise else None
    parallel_workers = int(train_args.get("parallel_workers", 1))
    effective_train_args = dict(train_args)

    def _run_sequential_layers(local_train_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], int]:
        for layer_idx in tqdm(
            sorted_layers,
            desc=f"Layers<{target_name}>",
            dynamic_ncols=True,
        ):
            refs = refs_by_layer[layer_idx]
            if not refs:
                logger.warning("Layer %d has no refs, skipped.", layer_idx)
                continue

            if data_type == "weight":
                dataset = WeightReferenceDataset(refs)
            else:
                dataset = ActivationReferenceDataset(refs)
            key = f"{target_name}_layer_{layer_idx}"
            rotation, metrics = train_one_rotation(
                dataset=dataset,
                hidden_size=hidden_size,
                train_args=local_train_args,
                device=device,
                logger=logger,
                train_name=key,
                head_num=head_num,
                wandb_run=wandb_run,
                data_type=data_type,
            )

            if is_head_wise:
                rotation_dict[key] = [rotation[i] for i in range(rotation.shape[0])]
            else:
                rotation_dict[key] = rotation
            metrics_dict[key] = metrics
        return rotation_dict, metrics_dict, hidden_size

    if parallel_workers > 1 and wandb_run is not None:
        logger.warning(
            "parallel_workers>1 with wandb is not supported for layer/head-wise mode; fallback to sequential."
        )
        parallel_workers = 1

    if parallel_workers > 1 and device.type == "cuda" and torch.cuda.device_count() <= 1:
        logger.warning(
            "parallel_workers=%d on a single CUDA device usually hurts throughput; forcing sequential.",
            parallel_workers,
        )
        parallel_workers = 1

    if parallel_workers > 1 and int(effective_train_args.get("num_workers", 0)) > 0:
        logger.warning(
            "parallel_workers=%d with DataLoader num_workers=%d may spawn too many processes; forcing num_workers=0.",
            parallel_workers,
            int(effective_train_args["num_workers"]),
        )
        effective_train_args["num_workers"] = 0

    if parallel_workers <= 1 or len(sorted_layers) <= 1:
        return _run_sequential_layers(effective_train_args)

    if device.type == "cuda":
        logger.warning(
            "Parallel layer/head-wise training on a single CUDA device may cause contention; monitor throughput."
        )

    worker_count = min(parallel_workers, len(sorted_layers))
    logger.info(
        "[%s] layer/head-wise parallel training enabled: workers=%d, layers=%d",
        target_name,
        worker_count,
        len(sorted_layers),
    )

    base_seed = int(getattr(args, "seed", 0))
    mp_context = mp.get_context("spawn")
    futures = {}
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            for layer_idx in sorted_layers:
                refs = refs_by_layer[layer_idx]
                if not refs:
                    logger.warning("Layer %d has no refs, skipped.", layer_idx)
                    continue
                future = executor.submit(
                    _train_single_layer_task,
                    layer_idx=int(layer_idx),
                    refs=refs,
                    target_name=target_name,
                    hidden_size=hidden_size,
                    train_args=effective_train_args,
                    device_arg=str(device),
                    head_num=head_num,
                    seed=base_seed,
                    data_type=data_type,
                )
                futures[future] = int(layer_idx)

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Layers<{target_name}> parallel",
                dynamic_ncols=True,
            ):
                layer_idx = futures[future]
                try:
                    done_layer_idx, rotation, metrics = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Layer {layer_idx} training failed in parallel worker.") from exc

                key = f"{target_name}_layer_{done_layer_idx}"
                if is_head_wise:
                    rotation_dict[key] = [rotation[i] for i in range(rotation.shape[0])]
                else:
                    rotation_dict[key] = rotation
                metrics_dict[key] = metrics
    except (PermissionError, OSError) as exc:
        logger.warning(
            "Failed to start parallel workers (%s). Falling back to sequential layer training.",
            exc,
        )
        return _run_sequential_layers(effective_train_args)

    return rotation_dict, metrics_dict, hidden_size


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_arg}")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    return torch.device(device_arg)
