#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_hadamard_utils():
    fake_quant_dir = _repo_root() / "fake_quant"
    sys.path.insert(0, str(fake_quant_dir))
    import hadamard_utils  # type: ignore

    return hadamard_utils


def _load_r1(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"R1 file not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(obj):
        r1 = obj
    elif isinstance(obj, dict):
        for key in ("R1", "r1", "rotate"):
            if key in obj and torch.is_tensor(obj[key]):
                r1 = obj[key]
                break
        else:
            raise KeyError(f"Could not find tensor key among ['R1','r1','rotate'] in: {path}")
    else:
        raise TypeError(f"Unsupported torch.load() type: {type(obj)} from {path}")

    if r1.ndim != 2 or r1.shape[0] != r1.shape[1]:
        raise ValueError(f"R1 must be square 2D, got shape={tuple(r1.shape)} from {path}")
    return r1


def _available_sample_ids(origin_dir: Path, layer: int, module: str) -> List[int]:
    if not origin_dir.exists():
        raise FileNotFoundError(f"origin_dir not found: {origin_dir}")
    pattern = re.compile(rf"^sample_(\d+)_layer_{int(layer)}_{re.escape(module)}\.pt$")
    ids: Set[int] = set()
    for name in os.listdir(origin_dir):
        match = pattern.match(name)
        if match is None:
            continue
        ids.add(int(match.group(1)))
    return sorted(ids)


def _choose_sample_ids(
    available: Sequence[int],
    num_samples: int,
    sample_ids: Optional[Sequence[int]],
    random_samples: bool,
    seed: int,
) -> List[int]:
    available_set = set(int(x) for x in available)

    if sample_ids is not None and len(sample_ids) > 0:
        chosen = [int(x) for x in sample_ids]
        missing = [x for x in chosen if x not in available_set]
        if missing:
            raise ValueError(f"Requested sample_ids not found for this layer/module: {missing[:20]}")
        if len(chosen) != num_samples:
            raise ValueError(f"--sample_ids has {len(chosen)} ids but --num_samples={num_samples}")
        return chosen

    if num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if len(available) < num_samples:
        raise ValueError(f"Not enough samples available ({len(available)}) to pick num_samples={num_samples}")

    if random_samples:
        rng = random.Random(seed)
        return sorted(rng.sample(list(available), num_samples))
    return list(available[:num_samples])


def _load_origin_activations(
    origin_dir: Path,
    layer: int,
    module: str,
    sample_ids: Sequence[int],
    tokens_per_sample: Optional[int],
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    xs: List[torch.Tensor] = []
    hidden_size: Optional[int] = None
    for sid in sample_ids:
        p = origin_dir / f"sample_{sid}_layer_{layer}_{module}.pt"
        x = torch.load(p, map_location="cpu", weights_only=True)
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor from {p}, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D activation tensor from {p}, got shape={tuple(x.shape)}")
        if hidden_size is None:
            hidden_size = int(x.shape[1])
        if int(x.shape[1]) != int(hidden_size):
            raise ValueError(f"Hidden size mismatch: got {int(x.shape[1])} from {p}, expected {hidden_size}")

        if tokens_per_sample is not None:
            tps = int(tokens_per_sample)
            if tps <= 0:
                raise ValueError("--tokens_per_sample must be > 0")
            if tps < int(x.shape[0]):
                idx = torch.tensor(sorted(rng.sample(range(int(x.shape[0])), tps)), dtype=torch.long)
                x = x.index_select(0, idx)
        xs.append(x)

    return torch.cat(xs, dim=0)


def _compute_curves(x: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    x: (N, H) float32 tensor on GPU/CPU.
    Returns 1D numpy arrays length H.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x.ndim==2, got shape={tuple(x.shape)}")

    x = x.float()
    qs = torch.tensor([0.0001, 0.01, 0.25, 0.75, 0.99, 0.9999], device=x.device, dtype=torch.float32)

    pmin = torch.amin(x, dim=0)
    pmax = torch.amax(x, dim=0)
    qv = torch.quantile(x, qs, dim=0)
    p0001, p01, p25, p75, p99, p9999 = qv.unbind(dim=0)

    out = {
        "pmin": pmin.detach().cpu().numpy(),
        "p0001": p0001.detach().cpu().numpy(),
        "p01": p01.detach().cpu().numpy(),
        "p25": p25.detach().cpu().numpy(),
        "p75": p75.detach().cpu().numpy(),
        "p99": p99.detach().cpu().numpy(),
        "p9999": p9999.detach().cpu().numpy(),
        "pmax": pmax.detach().cpu().numpy(),
    }
    return out


def _plot_compare(
    out_path: Path,
    before: Dict[str, np.ndarray],
    after: Dict[str, np.ndarray],
    title_left: str,
    title_right: str,
    suptitle: str,
    dpi: int,
    ylim_mode: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["p75", "p25", "p99", "p01", "p9999", "p0001", "pmin", "pmax"]
    colors = {
        "p75": "orange",
        "p25": "orange",
        "p99": "purple",
        "p01": "purple",
        "p9999": "red",
        "p0001": "red",
        "pmin": "blue",
        "pmax": "blue",
    }
    labels = {
        "p75": "25/75 Percentile",
        "p25": None,
        "p99": "1/99 Percentile",
        "p01": None,
        "p9999": "1/9999 Percentile",
        "p0001": None,
        "pmin": "Min/Max",
        "pmax": None,
    }

    h = len(before["pmin"])
    x = np.arange(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, curves, title in [(axes[0], before, title_left), (axes[1], after, title_right)]:
        for k in keys:
            ax.plot(x, curves[k], color=colors[k], label=labels[k], linewidth=0.3)
        ax.set_xlabel("Hidden dimension index", fontsize=14)
        ax.set_ylabel("Activation value", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)

    if ylim_mode != "auto":
        if ylim_mode == "minmax":
            y0 = min(float(before["pmin"].min()), float(after["pmin"].min()))
            y1 = max(float(before["pmax"].max()), float(after["pmax"].max()))
        elif ylim_mode == "p0001_p9999":
            y0 = min(float(before["p0001"].min()), float(after["p0001"].min()))
            y1 = max(float(before["p9999"].max()), float(after["p9999"].max()))
        elif ylim_mode == "p01_p99":
            y0 = min(float(before["p01"].min()), float(after["p01"].min()))
            y1 = max(float(before["p99"].max()), float(after["p99"].max()))
        else:
            raise ValueError(f"Unknown --ylim_mode: {ylim_mode}")
        if y0 == y1:
            y0, y1 = y0 - 1.0, y1 + 1.0
        pad = 0.02 * (y1 - y0)
        axes[0].set_ylim(y0 - pad, y1 + pad)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_quantile_grid4(
    out_path: Path,
    curves_by_name: Dict[str, Dict[str, np.ndarray]],
    suptitle: str,
    dpi: int,
    ylim_mode: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["p75", "p25", "p99", "p01", "p9999", "p0001", "pmin", "pmax"]
    colors = {
        "p75": "orange",
        "p25": "orange",
        "p99": "purple",
        "p01": "purple",
        "p9999": "red",
        "p0001": "red",
        "pmin": "blue",
        "pmax": "blue",
    }
    labels = {
        "p75": "25/75 Percentile",
        "p25": None,
        "p99": "1/99 Percentile",
        "p01": None,
        "p9999": "1/9999 Percentile",
        "p0001": None,
        "pmin": "Min/Max",
        "pmax": None,
    }

    order = ["origin", "hadamard", "spinquant", "dicequant"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0), sharex=True, sharey=False, constrained_layout=True)

    for ax, name in zip(axes, order, strict=True):
        curves = curves_by_name.get(name)
        if curves is None:
            ax.set_axis_off()
            continue

        h = len(curves["pmin"])
        x = np.arange(h)
        for k in keys:
            ax.plot(x, curves[k], color=colors[k], label=labels[k], linewidth=0.3)

        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Hidden dimension index", fontsize=12)
        ax.set_ylabel("Activation value", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

        if ylim_mode != "auto":
            if ylim_mode == "minmax":
                y0 = float(curves["pmin"].min())
                y1 = float(curves["pmax"].max())
            elif ylim_mode == "p0001_p9999":
                y0 = float(curves["p0001"].min())
                y1 = float(curves["p9999"].max())
            elif ylim_mode == "p01_p99":
                y0 = float(curves["p01"].min())
                y1 = float(curves["p99"].max())
            else:
                raise ValueError(f"Unknown --ylim_mode: {ylim_mode}")
            if y0 == y1:
                y0, y1 = y0 - 1.0, y1 + 1.0
            pad = 0.02 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)

    fig.suptitle(suptitle, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantile-curve compare plot for a specific layer/module activation using multiple samples."
    )
    parser.add_argument(
        "--origin_dir",
        type=Path,
        default=Path("/data02/cheny/dartquant/wikitext2_1024samples/llama3-8b-hf/r1_train"),
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer id, e.g. 1")
    parser.add_argument("--module", type=str, required=True, help="Module name in filename, e.g. mlp_up_proj")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to use as origin activation.")
    parser.add_argument("--sample_ids", type=int, nargs="*", default=None, help="Explicit sample ids (must match num_samples).")
    parser.add_argument("--random_samples", action="store_true", help="Randomly pick sample ids instead of the smallest ids.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokens_per_sample",
        type=int,
        default=None,
        help="If set, randomly sample this many token rows per sample (faster). Default uses all rows.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--layout",
        type=str,
        default="grid4",
        choices=["grid4", "compare"],
        help="grid4: 4 panels (origin/hadamard/spinquant/dicequant), y-axis not shared. compare: origin vs each method (2 panels).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=["hadamard", "spinquant", "dicequant"],
        choices=["hadamard", "spinquant", "dicequant"],
    )
    parser.add_argument(
        "--spinquant_r1",
        type=Path,
        default=_repo_root() / "spinquant_W16A4KV4_R.bin",
    )
    parser.add_argument(
        "--dicequant_r1",
        type=Path,
        default=Path(
            "/data02/cheny/dartquant/trained_rotation/"
            "wikitext2_1024samples/llama3-8b-hf/r1/"
            "sgd.0.0015.0.9.10.64.0.1.1/sgd.0.0015.0.9.10.64.0.1.1.pt"
        ),
    )
    parser.add_argument(
        "--ylim_mode",
        type=str,
        default="minmax",
        choices=["minmax", "p0001_p9999", "p01_p99", "auto"],
        help="compare: shared y-limits for the two subplots. grid4: applied per panel.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out_dir", type=Path, default=_repo_root() / "plot" / "layer_quantile_compare")
    args = parser.parse_args()

    hadamard_utils = _import_hadamard_utils()

    available = _available_sample_ids(args.origin_dir, args.layer, args.module)
    if not available:
        raise RuntimeError(f"No activation files found for layer={args.layer}, module={args.module} under {args.origin_dir}")

    chosen_ids = _choose_sample_ids(
        available,
        num_samples=args.num_samples,
        sample_ids=args.sample_ids,
        random_samples=args.random_samples,
        seed=args.seed,
    )

    x_cpu = _load_origin_activations(
        args.origin_dir,
        layer=args.layer,
        module=args.module,
        sample_ids=chosen_ids,
        tokens_per_sample=args.tokens_per_sample,
        seed=args.seed,
    )

    device = torch.device(args.device)
    x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

    # Load rotations only if needed
    spin_r1: Optional[torch.Tensor] = None
    dice_r1: Optional[torch.Tensor] = None
    if "spinquant" in args.methods:
        spin_r1 = _load_r1(args.spinquant_r1).to(dtype=torch.float32, device=device)
    if "dicequant" in args.methods:
        dice_r1 = _load_r1(args.dicequant_r1).to(dtype=torch.float32, device=device)

    # Hadamard config
    hidden_size = int(x.shape[1])
    had_K, K = hadamard_utils.get_hadK(hidden_size)
    had_K_dev = had_K.to(device=device, dtype=torch.float32) if had_K is not None else None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        origin_curves = _compute_curves(x.abs())

        layer_dir = args.out_dir / f"layer_{args.layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        tps_part = "all" if args.tokens_per_sample is None else f"{int(args.tokens_per_sample)}"

        meta = {
            "origin_dir": str(args.origin_dir),
            "layer": int(args.layer),
            "module": args.module,
            "chosen_sample_ids": [int(s) for s in chosen_ids],
            "num_samples": int(args.num_samples),
            "tokens_per_sample": None if args.tokens_per_sample is None else int(args.tokens_per_sample),
            "hidden_size": int(hidden_size),
            "methods": list(args.methods),
            "ylim_mode": args.ylim_mode,
            "layout": args.layout,
        }

        if args.layout == "compare":
            for method in args.methods:
                if method == "hadamard":
                    y = (
                        hadamard_utils.matmul_hadU_cuda(x, had_K_dev, K)
                        if device.type == "cuda"
                        else hadamard_utils.matmul_hadU(x)
                    )
                    title_right = "hadamard"
                elif method == "spinquant":
                    if spin_r1 is None:
                        raise RuntimeError("spin_r1 not loaded")
                    y = torch.matmul(x, spin_r1)
                    title_right = "spinquant (R1)"
                elif method == "dicequant":
                    if dice_r1 is None:
                        raise RuntimeError("dice_r1 not loaded")
                    y = torch.matmul(x, dice_r1)
                    title_right = "dicequant (R1)"
                else:
                    raise ValueError(f"Unknown method: {method}")

                after_curves = _compute_curves(y)
                del y
                if device.type == "cuda":
                    torch.cuda.synchronize()

                out_path = layer_dir / f"{args.module}_origin_vs_{method}_ns{args.num_samples}_tps{tps_part}.png"
                suptitle = f"layer={args.layer} module={args.module} | nsamples={args.num_samples} | ids={chosen_ids}"
                _plot_compare(
                    out_path,
                    before=origin_curves,
                    after=after_curves,
                    title_left="origin",
                    title_right=title_right,
                    suptitle=suptitle,
                    dpi=args.dpi,
                    ylim_mode=args.ylim_mode,
                )
                print(f"Saved: {out_path}")
        elif args.layout == "grid4":
            curves_by_name: Dict[str, Dict[str, np.ndarray]] = {"origin": origin_curves}

            if "hadamard" in args.methods:
                y = (
                    hadamard_utils.matmul_hadU_cuda(x, had_K_dev, K)
                    if device.type == "cuda"
                    else hadamard_utils.matmul_hadU(x)
                )
                curves_by_name["hadamard"] = _compute_curves(y.abs())
                del y
                if device.type == "cuda":
                    torch.cuda.synchronize()

            if "spinquant" in args.methods:
                if spin_r1 is None:
                    raise RuntimeError("spin_r1 not loaded")
                y = torch.matmul(x, spin_r1)
                curves_by_name["spinquant"] = _compute_curves(y.abs())
                del y
                if device.type == "cuda":
                    torch.cuda.synchronize()

            if "dicequant" in args.methods:
                if dice_r1 is None:
                    raise RuntimeError("dice_r1 not loaded")
                y = torch.matmul(x, dice_r1)
                curves_by_name["dicequant"] = _compute_curves(y.abs())
                del y
                if device.type == "cuda":
                    torch.cuda.synchronize()

            out_path = layer_dir / f"{args.module}_quantile4_ns{args.num_samples}_tps{tps_part}.png"
            suptitle = f"layer={args.layer} module={args.module} | nsamples={args.num_samples} | ids={chosen_ids}"
            _plot_quantile_grid4(
                out_path,
                curves_by_name=curves_by_name,
                suptitle=suptitle,
                dpi=args.dpi,
                ylim_mode=args.ylim_mode,
            )
            print(f"Saved: {out_path}")
        else:
            raise ValueError(f"Unknown --layout: {args.layout}")

    (args.out_dir / f"layer_{args.layer}" / f"{args.module}_{args.layout}_ns{args.num_samples}_tps{tps_part}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
