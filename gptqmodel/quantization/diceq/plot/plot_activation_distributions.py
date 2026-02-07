#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


FILENAME_RE = re.compile(r"sample_(\d+)_layer_(\d+)_([A-Za-z0-9_]+)\.pt$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_hadamard_utils():
    fake_quant_dir = _repo_root() / "fake_quant"
    sys.path.insert(0, str(fake_quant_dir))
    import hadamard_utils  # type: ignore

    return hadamard_utils


def _parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}. Choices: {sorted(mapping)}")
    return mapping[dtype]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _list_activation_files(
    origin_dir: Path,
    modules: Optional[Sequence[str]],
    layers: Optional[Sequence[int]],
    nsamples: Optional[int],
) -> List[Path]:
    if not origin_dir.exists():
        raise FileNotFoundError(f"origin_dir not found: {origin_dir}")

    modules_set = set(modules) if modules else None
    layers_set = set(int(x) for x in layers) if layers else None
    nsamples_int = int(nsamples) if nsamples is not None else None

    out: List[Path] = []
    for name in os.listdir(origin_dir):
        match = FILENAME_RE.match(name)
        if match is None:
            continue
        sample_id = int(match.group(1))
        layer_id = int(match.group(2))
        module_name = match.group(3)

        if nsamples_int is not None and sample_id >= nsamples_int:
            continue
        if layers_set is not None and layer_id not in layers_set:
            continue
        if modules_set is not None and module_name not in modules_set:
            continue
        out.append(origin_dir / name)

    out.sort()
    return out


def _balance_and_sample_files(
    files: Sequence[Path],
    max_files: int,
    seed: int,
    balance_by_module: bool,
) -> List[Path]:
    if max_files <= 0:
        raise ValueError("--max_files must be > 0")
    if max_files >= len(files):
        return list(files)

    rng = random.Random(seed)
    if not balance_by_module:
        files_shuffled = list(files)
        rng.shuffle(files_shuffled)
        return files_shuffled[: min(max_files, len(files_shuffled))]

    by_module: Dict[str, List[Path]] = {}
    for p in files:
        match = FILENAME_RE.match(p.name)
        if match is None:
            continue
        module_name = match.group(3)
        by_module.setdefault(module_name, []).append(p)

    if not by_module:
        return []

    modules = sorted(by_module.keys())
    per = max_files // len(modules)
    rem = max_files % len(modules)

    picked: List[Path] = []
    for i, module in enumerate(modules):
        bucket = list(by_module[module])
        rng.shuffle(bucket)
        take = per + (1 if i < rem else 0)
        picked.extend(bucket[: min(take, len(bucket))])

    rng.shuffle(picked)
    return picked[: min(max_files, len(picked))]


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


def _sample_rows(x_cpu: torch.Tensor, rows_per_file: int, rng: random.Random) -> torch.Tensor:
    if x_cpu.ndim != 2:
        raise ValueError(f"Expected activation tensor with 2 dims, got shape={tuple(x_cpu.shape)}")

    n_rows = x_cpu.shape[0]
    if rows_per_file < 0:
        raise ValueError("--rows_per_file must be >= 0 (0 means all rows)")
    if rows_per_file == 0:
        return x_cpu
    if rows_per_file >= n_rows:
        return x_cpu

    idx = torch.tensor(rng.sample(range(n_rows), rows_per_file), dtype=torch.long)
    return x_cpu.index_select(0, idx)


def _hadamard_transform(
    x: torch.Tensor,
    hadamard_utils,
    had_K: Optional[torch.Tensor],
    K: int,
) -> torch.Tensor:
    if x.device.type == "cuda":
        return hadamard_utils.matmul_hadU_cuda(x, had_K, K)
    return hadamard_utils.matmul_hadU(x)


def _summarize(values: np.ndarray) -> Dict[str, float]:
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "abs_p99": float(np.percentile(abs_values, 99.0)),
        "abs_p999": float(np.percentile(abs_values, 99.9)),
        "abs_max": float(abs_values.max()),
    }


def _summarize_from_moments(
    count: int,
    sum_: float,
    sumsq: float,
    min_v: float,
    max_v: float,
    abs_max: float,
) -> Dict[str, float | None]:
    if count <= 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_max": 0.0,
            "abs_p99": None,
            "abs_p999": None,
        }
    mean = sum_ / count
    var = max(0.0, (sumsq / count) - mean * mean)
    return {
        "count": int(count),
        "mean": float(mean),
        "std": float(var**0.5),
        "min": float(min_v),
        "max": float(max_v),
        "abs_max": float(abs_max),
        "abs_p99": None,
        "abs_p999": None,
    }


def _palette_colors(palette: str) -> Dict[str, str]:
    pal = str(palette).lower()
    if pal in {"paper", "vega"}:
        # Matplotlib-default-like palette (matches common paper figures).
        return {
            "origin": "#1F77B4",
            "hadamard": "#FF7F0E",
            "spinquant": "#2CA02C",
            "dicequant": "#D62728",
        }
    if pal in {"okabe_ito", "okabe-ito", "colorblind"}:
        return {
            "origin": "#0072B2",
            "hadamard": "#E69F00",
            "spinquant": "#009E73",
            "dicequant": "#D55E00",
        }
    if pal in {"mono", "monochrome"}:
        return {
            "origin": "#5DA5DA",
            "hadamard": "#5DA5DA",
            "spinquant": "#5DA5DA",
            "dicequant": "#5DA5DA",
        }
    raise ValueError(f"Unknown --palette: {palette}. Choices: paper|okabe_ito|mono")


def _quant_levels_symmetric(scale: float, bits: int) -> np.ndarray:
    if bits <= 0:
        raise ValueError("--quant_bits must be > 0")
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax - 1
    qs = np.arange(int(qmin), int(qmax) + 1, dtype=np.float64)
    return qs * float(scale)


def _compute_grid4_streaming(
    picked_files: Sequence[Path],
    rows_per_file: int,
    rng: random.Random,
    device: torch.device,
    compute_dtype: torch.dtype,
    dim_idx_dev: Optional[torch.Tensor],
    spin_r1: torch.Tensor,
    dice_r1: torch.Tensor,
    hadamard_utils,
    had_K_dev: Optional[torch.Tensor],
    K: int,
    bins: int,
    quant_bits: Optional[int],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, float | None]],
    Optional[Dict[str, Dict[str, float]]],
]:
    order = ["origin", "hadamard", "spinquant", "dicequant"]

    mins = {k: float("inf") for k in order}
    maxs = {k: float("-inf") for k in order}
    abs_maxs = {k: 0.0 for k in order}
    sums = {k: 0.0 for k in order}
    sumsqs = {k: 0.0 for k in order}
    counts = {k: 0 for k in order}

    def update_moments(name: str, t: torch.Tensor) -> None:
        t32 = t.float()
        vec = torch.stack(
            [
                t32.min().to(torch.float64),
                t32.max().to(torch.float64),
                t32.abs().max().to(torch.float64),
                t32.sum(dtype=torch.float64),
                t32.square().sum(dtype=torch.float64),
            ]
        )
        vmin, vmax, vmaxabs, vsum, vsumsq = vec.detach().cpu().tolist()
        mins[name] = min(mins[name], float(vmin))
        maxs[name] = max(maxs[name], float(vmax))
        abs_maxs[name] = max(abs_maxs[name], float(vmaxabs))
        sums[name] += float(vsum)
        sumsqs[name] += float(vsumsq)
        counts[name] += int(t32.numel())

    # First pass: moments + min/max per method
    with torch.inference_mode():
        for i, p in enumerate(picked_files):
            x_cpu = torch.load(p, map_location="cpu", weights_only=True)
            if not torch.is_tensor(x_cpu):
                raise TypeError(f"Expected tensor from {p}, got {type(x_cpu)}")
            x_cpu = _sample_rows(x_cpu, rows_per_file, rng)
            x = x_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)

            x_origin = x.index_select(1, dim_idx_dev) if dim_idx_dev is not None else x
            x_had = _hadamard_transform(x, hadamard_utils, had_K_dev, K)
            if dim_idx_dev is not None:
                x_had = x_had.index_select(1, dim_idx_dev)
            x_spin = torch.matmul(x, spin_r1)
            x_dice = torch.matmul(x, dice_r1)

            update_moments("origin", x_origin)
            update_moments("hadamard", x_had)
            update_moments("spinquant", x_spin)
            update_moments("dicequant", x_dice)

            del x_origin, x_had, x_spin, x_dice, x
            if (i + 1) % 8 == 0 or (i + 1) == len(picked_files):
                print(f"[pass1 {i+1}/{len(picked_files)}] {p.name}")

    quant: Optional[Dict[str, Dict[str, float]]] = None
    scale_by_label: Dict[str, float] = {}
    if quant_bits is not None:
        qmax = (2 ** (int(quant_bits) - 1)) - 1
        if qmax <= 0:
            raise ValueError("--quant_bits must be >= 2 for signed symmetric quantization")
        for k in order:
            abs_max = float(abs_maxs[k])
            scale_by_label[k] = (abs_max / float(qmax)) if abs_max > 0 else 1.0

    edges_by_label: Dict[str, np.ndarray] = {}
    for k in order:
        lo, hi = mins[k], maxs[k]
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite min/max for {k}: min={lo} max={hi}")
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        edges_by_label[k] = np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)

    # Second pass: histogram counts per method (streaming)
    hist_counts = {k: np.zeros(int(bins), dtype=np.float64) for k in order}
    sse_dev: Optional[Dict[str, torch.Tensor]] = None
    mse_counts: Optional[Dict[str, int]] = None
    qmin: Optional[int] = None
    qmax: Optional[int] = None
    if quant_bits is not None:
        qmax = (2 ** (int(quant_bits) - 1)) - 1
        qmin = -qmax - 1
        sse_dev = {k: torch.zeros((), device=device, dtype=torch.float64) for k in order}
        mse_counts = {k: 0 for k in order}

    with torch.inference_mode():
        for i, p in enumerate(picked_files):
            x_cpu = torch.load(p, map_location="cpu", weights_only=True)
            x_cpu = _sample_rows(x_cpu, rows_per_file, rng)
            x = x_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)

            x_origin = x.index_select(1, dim_idx_dev) if dim_idx_dev is not None else x
            x_had = _hadamard_transform(x, hadamard_utils, had_K_dev, K)
            if dim_idx_dev is not None:
                x_had = x_had.index_select(1, dim_idx_dev)
            x_spin = torch.matmul(x, spin_r1)
            x_dice = torch.matmul(x, dice_r1)

            for name, t in [("origin", x_origin), ("hadamard", x_had), ("spinquant", x_spin), ("dicequant", x_dice)]:
                edges = edges_by_label[name]
                lo, hi = float(edges[0]), float(edges[-1])
                c = torch.histc(t.float(), bins=int(bins), min=lo, max=hi).detach().cpu().numpy().astype(np.float64)
                hist_counts[name] += c
                if sse_dev is not None and mse_counts is not None and qmin is not None and qmax is not None:
                    scale = float(scale_by_label[name])
                    scale_t = torch.tensor(scale, device=t.device, dtype=t.dtype)
                    qt = torch.clamp(torch.round(t / scale_t), qmin, qmax)
                    deq = qt * scale_t
                    err = (t - deq).float()
                    sse_dev[name] += err.square().sum(dtype=torch.float64)
                    mse_counts[name] += int(err.numel())

            del x_origin, x_had, x_spin, x_dice, x
            if (i + 1) % 8 == 0 or (i + 1) == len(picked_files):
                print(f"[pass2 {i+1}/{len(picked_files)}] {p.name}")

    hists_density: Dict[str, np.ndarray] = {}
    for k in order:
        edges = edges_by_label[k]
        bin_width = float(edges[1] - edges[0])
        total = float(hist_counts[k].sum())
        hists_density[k] = (hist_counts[k] / (total * bin_width)).astype(np.float64) if total > 0 else hist_counts[k]

    stats = {
        k: _summarize_from_moments(
            count=counts[k],
            sum_=sums[k],
            sumsq=sumsqs[k],
            min_v=mins[k],
            max_v=maxs[k],
            abs_max=abs_maxs[k],
        )
        for k in order
    }

    if sse_dev is not None and mse_counts is not None:
        mse_by_label: Dict[str, float] = {}
        for k in order:
            n = int(mse_counts[k])
            mse_by_label[k] = float((sse_dev[k].detach().cpu().item()) / n) if n > 0 else 0.0
        quant = {"scale_by_method": {k: float(scale_by_label[k]) for k in order}, "mse_by_method": mse_by_label}

    return edges_by_label, hists_density, stats, quant


def _plot_hist(
    out_path: Path,
    hists: Dict[str, np.ndarray],
    edges: np.ndarray,
    title: str,
    dpi: int,
    figsize: Tuple[float, float],
    font_scale: float,
    palette: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    centers = (edges[:-1] + edges[1:]) / 2

    label_fs = 12.0 * float(font_scale)
    tick_fs = 10.0 * float(font_scale)
    title_fs = 11.0 * float(font_scale)
    legend_fs = 9.0 * float(font_scale)

    colors = _palette_colors(palette)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for label, hist in hists.items():
        color = colors.get(label, "#4C78A8")
        line_color = mcolors.to_hex(tuple(np.array(mcolors.to_rgb(color)) * 0.85))
        ax1.plot(centers, hist, label=label, linewidth=2.2, color=line_color)
        ax2.plot(centers, np.maximum(hist, 1e-12), label=label, linewidth=2.2, color=line_color)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("activation value", fontsize=label_fs)
        ax.set_ylabel("density", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_xlim(edges[0], edges[-1])

    ax1.set_title("linear", fontsize=title_fs)
    ax2.set_title("log-y", fontsize=title_fs)
    ax2.set_yscale("log")
    fig.suptitle(title, fontsize=title_fs)
    ax1.legend(loc="upper right", fontsize=legend_fs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_hist_grid4(
    out_path: Path,
    hists: Dict[str, np.ndarray],
    edges_by_label: Dict[str, np.ndarray],
    title: str,
    logy: bool,
    smooth_sigma: float,
    show_title: bool,
    dpi: int,
    figsize: Tuple[float, float],
    font_scale: float,
    palette: str,
    quant_bits: Optional[int],
    quant_levels_by_label: Optional[Dict[str, np.ndarray]],
    mse_by_label: Optional[Dict[str, float]],
    quant_grid_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return y
        radius = max(1, int(3 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
        kernel /= kernel.sum()
        return np.convolve(y, kernel, mode="same")

    order = ["origin", "hadamard", "spinquant", "dicequant"]
    captions = {
        "origin": "(a) Original.",
        "hadamard": "(b) Hadamard.",
        "spinquant": "(c) SpinQuant.",
        "dicequant": "(d) DICEQuant.",
    }

    label_fs = 12.0 * float(font_scale)
    tick_fs = 10.0 * float(font_scale)
    title_fs = 11.0 * float(font_scale)
    caption_fs = 14.0 * float(font_scale)

    colors = _palette_colors(palette)

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=False, sharey=False)
    for ax, key in zip(axes, order, strict=True):
        edges = edges_by_label[key]
        centers = (edges[:-1] + edges[1:]) / 2
        hist = hists[key]
        y = np.maximum(hist, 1e-12) if logy else hist
        y = gaussian_smooth(y, smooth_sigma)
        color = colors.get(key, "#5DA5DA")
        line_color = mcolors.to_hex(tuple(np.array(mcolors.to_rgb(color)) * 0.85))
        baseline = 1e-12 if logy else 0.0
        ax.fill_between(centers, y, baseline, color=color, alpha=0.35, linewidth=0.0)
        ax.plot(centers, y, color=line_color, linewidth=2.4)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Activation Value", fontsize=label_fs)
        ax.set_ylabel("Probability Density", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_xlim(edges[0], edges[-1])
        if logy:
            ax.set_yscale("log")

        if quant_levels_by_label is not None and key in quant_levels_by_label:
            levels = quant_levels_by_label[key]
            levels = levels[(levels >= float(edges[0])) & (levels <= float(edges[-1]))]
            if levels.size > 0:
                ax.vlines(
                    levels,
                    0.0,
                    0.12,
                    transform=ax.get_xaxis_transform(),
                    color="#222222",
                    alpha=float(quant_grid_alpha),
                    linewidth=1.1,
                    zorder=3,
                )

        if mse_by_label is not None and key in mse_by_label and quant_bits is not None:
            mse = float(mse_by_label[key])
            ax.set_title(
                f"MSE={mse:.2e}",
                loc="center",
                fontsize=10.0 * float(font_scale),
                pad=3.0 * float(font_scale),
            )

        ax.text(
            0.5,
            -0.32,
            captions.get(key, key),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=caption_fs,
            fontweight="bold",
            family="serif",
        )

    if show_title:
        fig.suptitle(title, fontsize=title_fs)
    fig.subplots_adjust(bottom=0.3, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot activation distributions for origin / hadamard / SpinQuant(R1) / DICEQuant(R1)."
    )
    parser.add_argument(
        "--origin_dir",
        type=Path,
        default=Path("/data02/cheny/dartquant/wikitext2_1024samples/llama3-8b-hf/r1_train"),
        help="Directory with per-layer activation .pt files (sample_*_layer_*_*.pt).",
    )
    parser.add_argument(
        "--spinquant_r1",
        type=Path,
        default=_repo_root() / "spinquant_W16A4KV4_R.bin",
        help="SpinQuant checkpoint containing R1 (torch.load-able .bin).",
    )
    parser.add_argument(
        "--dicequant_r1",
        type=Path,
        default=Path(
            "/data02/cheny/dartquant/QSR-geoopt/trained_rotation/"
            "wikitext2_1024samples/llama3-8b-hf/r1/"
            "RAdam.0.1.0.9.40.128.0.3.1.cos/RAdam.0.1.0.9.40.128.0.3.1.cos.pt"
        ),
        help="DICEQuant checkpoint containing R1 (.pt saved dict with key 'R1').",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["self_attn_q_proj", "mlp_up_proj"],
        help="Filter module names in filenames. Default uses both.",
    )
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="Optional layer id filter (e.g. 0 1 2).")
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Only include sample_<id> with id < nsamples (default: include all in dir).",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Max activation files to load for plotting.")
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=-1,
        help="How many token rows to sample per file. -1 uses auto (grid4=all rows, overlay=32). 0 means all rows.",
    )
    parser.add_argument(
        "--ndims",
        type=int,
        default=None,
        help="Optional: sample ndims output dims for all methods (faster for rotation). Default uses all dims.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for file/row sampling.")
    parser.add_argument("--device", type=str, default=_default_device(), help="cuda or cpu (default: auto).")
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        help="float16|bfloat16|float32. Default: float16 on cuda, else float32.",
    )
    parser.add_argument("--bins", type=int, default=400, help="Histogram bins.")
    parser.add_argument(
        "--clip_pctl",
        type=float,
        default=99.9,
        help="(overlay) Global clip range by percentile of |x| over all collected values (default: 99.9).",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=None,
        help="If set, use fixed symmetric clip range [-clip_value, clip_value] (overrides --clip_pctl).",
    )
    parser.add_argument("--out_dir", type=Path, default=_repo_root() / "plot" / "activation_distributions")
    parser.add_argument("--out_name", type=str, default=None, help="Output filename (default: auto).")
    parser.add_argument(
        "--out_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format/extension (default: png). For papers, pdf/svg are recommended.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI for raster outputs (e.g. png).")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches. Default depends on --layout.",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.2,
        help="Scale plot font sizes (axis labels/ticks/captions).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="overlay",
        choices=["overlay", "grid4"],
        help="overlay: all methods overlaid (+log-y panel). grid4: 4 subplots side-by-side (independent y-axis).",
    )
    parser.add_argument("--logy", action="store_true", help="Use log y-axis for --layout grid4.")
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma (in histogram bins) for --layout grid4. 0 disables smoothing.",
    )
    parser.add_argument(
        "--grid4_title",
        action="store_true",
        help="Show a figure title for --layout grid4 (default: off to match paper-style figures).",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="paper",
        choices=["paper", "okabe_ito", "mono"],
        help="Color palette for plotting.",
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=None,
        help="If set (e.g. 4), overlay signed symmetric uniform quantization grid and compute MSE (grid4 only).",
    )
    parser.add_argument(
        "--quant_grid_alpha",
        type=float,
        default=0.55,
        help="Alpha for quantization grid ticks (grid4 only).",
    )
    parser.add_argument(
        "--balance_by_module",
        action="store_true",
        help="Sample roughly equal #files per module before max_files cut.",
    )
    args = parser.parse_args()

    rows_per_file = int(args.rows_per_file)
    if rows_per_file < 0:
        rows_per_file = 0 if args.layout == "grid4" else 32

    device = torch.device(args.device)
    compute_dtype = _parse_dtype(args.compute_dtype)

    hadamard_utils = _import_hadamard_utils()

    # Load rotation matrices
    spin_r1 = _load_r1(args.spinquant_r1).to(dtype=torch.float32)
    dice_r1 = _load_r1(args.dicequant_r1).to(dtype=torch.float32)

    if spin_r1.shape != dice_r1.shape:
        raise ValueError(f"SpinQuant R1 {tuple(spin_r1.shape)} != DICEQuant R1 {tuple(dice_r1.shape)}")
    hidden_size = int(spin_r1.shape[0])

    rng = random.Random(args.seed)
    dim_idx: Optional[torch.Tensor] = None
    if args.ndims is not None:
        ndims = int(args.ndims)
        if ndims <= 0:
            raise ValueError("--ndims must be > 0")
        if ndims < hidden_size:
            dim_idx = torch.tensor(sorted(rng.sample(range(hidden_size), ndims)), dtype=torch.long)
        elif ndims > hidden_size:
            raise ValueError(f"--ndims={ndims} > hidden_size={hidden_size}")

    spin_r1 = spin_r1.to(device=device, dtype=compute_dtype, non_blocking=True)
    dice_r1 = dice_r1.to(device=device, dtype=compute_dtype, non_blocking=True)
    dim_idx_dev: Optional[torch.Tensor] = None
    if dim_idx is not None:
        dim_idx_dev = dim_idx.to(device=device)
        spin_r1 = spin_r1.index_select(1, dim_idx_dev)
        dice_r1 = dice_r1.index_select(1, dim_idx_dev)

    # Hadamard config (depends on dim)
    had_K, K = hadamard_utils.get_hadK(hidden_size)
    if had_K is not None:
        had_K = had_K.to(dtype=torch.float32)
    had_K_dev = had_K.to(device=device, dtype=compute_dtype) if had_K is not None else None

    # Pick activation files
    all_files = _list_activation_files(args.origin_dir, args.modules, args.layers, args.nsamples)
    if not all_files:
        raise RuntimeError("No activation files matched filters.")

    if args.max_files is None:
        # If user explicitly selects layer(s) + sample range, default to using all matched files.
        if args.layers is not None and args.nsamples is not None:
            max_files = len(all_files)
        else:
            max_files = min(32, len(all_files))
    else:
        max_files = int(args.max_files)

    picked_files = _balance_and_sample_files(
        all_files, max_files=max_files, seed=args.seed, balance_by_module=args.balance_by_module
    )
    if not picked_files:
        raise RuntimeError("No activation files selected after sampling.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    quant_bits = int(args.quant_bits) if args.quant_bits is not None else None
    quant_levels_by_label: Optional[Dict[str, np.ndarray]] = None
    mse_by_label: Optional[Dict[str, float]] = None
    quant_meta: Optional[Dict[str, Dict[str, float]]] = None

    if quant_bits is not None and args.layout != "grid4":
        print("WARNING: --quant_bits currently only supports --layout grid4; ignoring quant overlay/MSE.")
        quant_bits = None

    if args.layout == "grid4" and rows_per_file == 0:
        edges_by_label, hists, stats, quant_meta = _compute_grid4_streaming(
            picked_files=picked_files,
            rows_per_file=rows_per_file,
            rng=rng,
            device=device,
            compute_dtype=compute_dtype,
            dim_idx_dev=dim_idx_dev,
            spin_r1=spin_r1,
            dice_r1=dice_r1,
            hadamard_utils=hadamard_utils,
            had_K_dev=had_K_dev,
            K=K,
            bins=int(args.bins),
            quant_bits=quant_bits,
        )
        clip = None
        clip_by_label = {k: max(abs(float(v[0])), abs(float(v[-1]))) for k, v in edges_by_label.items()}
        edges = None
    else:
        values: Dict[str, List[np.ndarray]] = {
            "origin": [],
            "hadamard": [],
            "spinquant": [],
            "dicequant": [],
        }

        with torch.inference_mode():
            for i, p in enumerate(picked_files):
                x_cpu = torch.load(p, map_location="cpu", weights_only=True)
                if not torch.is_tensor(x_cpu):
                    raise TypeError(f"Expected tensor from {p}, got {type(x_cpu)}")
                if x_cpu.ndim != 2 or x_cpu.shape[-1] != hidden_size:
                    raise ValueError(f"Unexpected activation shape from {p}: {tuple(x_cpu.shape)}")

                x_cpu = _sample_rows(x_cpu, rows_per_file, rng)
                x = x_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)

                x_origin = x.index_select(1, dim_idx_dev) if dim_idx_dev is not None else x
                x_had = _hadamard_transform(x, hadamard_utils, had_K_dev, K)
                if dim_idx_dev is not None:
                    x_had = x_had.index_select(1, dim_idx_dev)
                x_spin = torch.matmul(x, spin_r1)
                x_dice = torch.matmul(x, dice_r1)

                values["origin"].append(x_origin.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy())
                values["hadamard"].append(x_had.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy())
                values["spinquant"].append(x_spin.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy())
                values["dicequant"].append(x_dice.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy())

                if (i + 1) % 8 == 0 or (i + 1) == len(picked_files):
                    print(f"[{i+1}/{len(picked_files)}] processed {p.name}")

        merged: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in values.items()}
        stats = {k: _summarize(arr) for k, arr in merged.items()}

        if args.layout == "overlay":
            if args.clip_value is not None:
                clip = float(args.clip_value)
            else:
                all_abs = np.concatenate([np.abs(v) for v in merged.values()], axis=0)
                clip = float(np.percentile(all_abs, float(args.clip_pctl)))
                clip = max(clip, 1e-6)
            edges = np.linspace(-clip, clip, int(args.bins) + 1, dtype=np.float64)
            edges_by_label = {k: edges for k in merged.keys()}
            hists = {k: np.histogram(v, bins=edges, density=True)[0] for k, v in merged.items()}
            clip_by_label = {k: clip for k in merged.keys()}
        elif args.layout == "grid4":
            edges_by_label = {}
            hists = {}
            clip_by_label = {}
            clip = float(args.clip_value) if args.clip_value is not None else None
            for k, v in merged.items():
                if args.clip_value is not None:
                    lo, hi = -float(args.clip_value), float(args.clip_value)
                else:
                    lo, hi = float(np.min(v)), float(np.max(v))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError(f"Non-finite values in {k}: min={lo} max={hi}")
                    if lo == hi:
                        lo, hi = lo - 1e-6, hi + 1e-6
                edges_k = np.linspace(lo, hi, int(args.bins) + 1, dtype=np.float64)
                edges_by_label[k] = edges_k
                hists[k] = np.histogram(v, bins=edges_k, density=True)[0]
                clip_by_label[k] = max(abs(lo), abs(hi))
            edges = None

            if quant_bits is not None:
                qmax = (2 ** (int(quant_bits) - 1)) - 1
                qmin = -qmax - 1
                scale_by_method: Dict[str, float] = {}
                mse_by_method: Dict[str, float] = {}
                for k, v in merged.items():
                    abs_max = float(np.max(np.abs(v)))
                    scale = (abs_max / float(qmax)) if abs_max > 0 else 1.0
                    scale_by_method[k] = float(scale)
                    q = np.clip(np.round(v / scale), qmin, qmax)
                    deq = q * scale
                    mse_by_method[k] = float(np.mean((v - deq) ** 2))
                quant_meta = {"scale_by_method": scale_by_method, "mse_by_method": mse_by_method}
        else:
            raise ValueError(f"Unknown layout: {args.layout}")

    if quant_meta is not None and quant_bits is not None:
        mse_by_label = {k: float(quant_meta["mse_by_method"][k]) for k in quant_meta["mse_by_method"].keys()}
        qmax = (2 ** (int(quant_bits) - 1)) - 1
        scale_by = quant_meta.get("scale_by_method", {})
        quant_levels_by_label = {k: _quant_levels_symmetric(float(scale_by[k]), int(quant_bits)) for k in scale_by.keys()}

    modules_part = "+".join(args.modules) if args.modules else "all"
    layers_part = "all" if args.layers is None else "+".join(str(x) for x in args.layers)
    ndims_part = "all" if args.ndims is None else str(int(args.ndims))
    rows_part = "all" if rows_per_file == 0 else str(rows_per_file)
    out_ext = str(args.out_format).lstrip(".").lower()
    auto_name = (
        f"act_dist_modules-{modules_part}_layers-{layers_part}"
        f"_files{len(picked_files)}_rows{rows_part}_ndims{ndims_part}_layout{args.layout}.{out_ext}"
    )
    if args.out_name is None:
        out_name = auto_name
    else:
        out_name = str(args.out_name)
        if Path(out_name).suffix == "":
            out_name = out_name + f".{out_ext}"
    out_path = args.out_dir / out_name

    title = (
        "Activation distribution (origin / hadamard / spinquant R1 / dicequant R1)\n"
        f"origin_dir={args.origin_dir} | files={len(picked_files)} | rows/file={rows_part} | ndims={ndims_part}"
    )
    if args.layout == "overlay":
        figsize = (13.0, 4.2) if args.figsize is None else (float(args.figsize[0]), float(args.figsize[1]))
        _plot_hist(
            out_path,
            hists,
            edges,
            title,
            dpi=int(args.dpi),
            figsize=figsize,
            font_scale=float(args.font_scale),
            palette=str(args.palette),
        )
    elif args.layout == "grid4":
        figsize = (18.0, 4.2) if args.figsize is None else (float(args.figsize[0]), float(args.figsize[1]))
        _plot_hist_grid4(
            out_path,
            hists,
            edges_by_label,
            title,
            logy=bool(args.logy),
            smooth_sigma=float(args.smooth_sigma),
            show_title=bool(args.grid4_title),
            dpi=int(args.dpi),
            figsize=figsize,
            font_scale=float(args.font_scale),
            palette=str(args.palette),
            quant_bits=quant_bits,
            quant_levels_by_label=quant_levels_by_label,
            mse_by_label=mse_by_label,
            quant_grid_alpha=float(args.quant_grid_alpha),
        )
    else:
        raise ValueError(f"Unknown layout: {args.layout}")

    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta = {
        "args": args_json,
        "picked_files": [str(p) for p in picked_files],
        "stats": stats,
        "clip": clip,
        "clip_by_method": clip_by_label,
    }
    if quant_bits is not None and quant_meta is not None:
        meta["quant"] = {
            "bits": int(quant_bits),
            "scheme": "symmetric_uniform",
            "scale_method": "absmax",
            **quant_meta,
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / (Path(out_name).stem + ".summary.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.layout == "overlay":
        np.savez_compressed(args.out_dir / (Path(out_name).stem + ".hist.npz"), edges=edges, **hists)
    else:
        np.savez_compressed(
            args.out_dir / (Path(out_name).stem + ".hist.npz"),
            **{f"edges_{k}": v for k, v in edges_by_label.items()},
            **hists,
        )

    print(f"Saved: {out_path}")
    print(f"Saved: {args.out_dir / (Path(out_name).stem + '.summary.json')}")
    print(f"Saved: {args.out_dir / (Path(out_name).stem + '.hist.npz')}")


if __name__ == "__main__":
    main()
