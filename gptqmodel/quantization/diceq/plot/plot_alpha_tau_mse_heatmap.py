#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


DEMO_TSV = """alpha\ttau\tmse
5\t0.005\t0.098309
10\t0.005\t0.07896298
20\t0.005\t0.063936537
40\t0.005\t0.054920672
80\t0.005\t0.051915383
160\t0.005\t0.054920672
5\t0.01\t0.083198281
10\t0.01\t0.066845882
20\t0.01\t0.055165597
40\t0.01\t0.048157426
80\t0.01\t0.045821369
160\t0.01\t0.048157426
5\t0.02\t0.072350548
10\t0.02\t0.059022691
20\t0.02\t0.049502794
40\t0.02\t0.043790855
80\t0.02\t0.041886875
160\t0.02\t0.043790855
5\t0.05\t0.067148272
10\t0.05\t0.055270903
20\t0.05\t0.046787068
40\t0.05\t0.041696767
80\t0.05\t0.04097
160\t0.05\t0.041696767
5\t0.1\t0.070125265
10\t0.1\t0.057417857
20\t0.1\t0.048341137
40\t0.1\t0.042895105
80\t0.1\t0.041079761
160\t0.1\t0.042895105
5\t0.2\t0.079056243
10\t0.2\t0.063858718
20\t0.2\t0.053003343
40\t0.2\t0.046490118
80\t0.2\t0.044319043
160\t0.2\t0.046490118
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text_input(path: str) -> str:
    if path == "-" or path == "/dev/stdin":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _load_table(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = text.strip()
    if not s:
        raise ValueError("Empty input.")

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Need at least a header line and one data line.")

    header = lines[0]
    if "," in header:
        delim = ","
        split = lambda ln: [x.strip() for x in ln.split(delim)]
    elif "\t" in header:
        delim = "\t"
        split = lambda ln: [x.strip() for x in ln.split(delim)]
    else:
        split = lambda ln: ln.split()

    cols = [c.strip().lower() for c in split(header)]
    want = {}
    for name in ("alpha", "tau", "mse"):
        if name not in cols:
            raise ValueError(f"Missing column '{name}'. Got columns: {cols}")
        want[name] = cols.index(name)

    alpha: list[float] = []
    tau: list[float] = []
    mse: list[float] = []
    for ln in lines[1:]:
        parts = split(ln)
        if len(parts) < len(cols):
            raise ValueError(f"Bad row (expected {len(cols)} cols): {ln}")
        alpha.append(float(parts[want["alpha"]]))
        tau.append(float(parts[want["tau"]]))
        mse.append(float(parts[want["mse"]]))

    if not alpha:
        raise ValueError("No rows found.")
    return np.asarray(alpha, dtype=np.float64), np.asarray(tau, dtype=np.float64), np.asarray(mse, dtype=np.float64)


def _make_matrix(
    alpha: np.ndarray,
    tau: np.ndarray,
    mse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.unique(alpha)
    taus = np.unique(tau)
    alphas.sort()
    taus.sort()

    mat = np.full((alphas.size, taus.size), np.nan, dtype=np.float64)
    a_to_i = {float(a): i for i, a in enumerate(alphas.tolist())}
    t_to_j = {float(t): j for j, t in enumerate(taus.tolist())}
    for a, t, v in zip(alpha.tolist(), tau.tolist(), mse.tolist(), strict=True):
        i = a_to_i[float(a)]
        j = t_to_j[float(t)]
        if np.isfinite(mat[i, j]):
            raise ValueError(f"Duplicate entry for alpha={a}, tau={t}.")
        mat[i, j] = float(v)

    if not np.all(np.isfinite(mat)):
        missing = int(np.size(mat) - np.isfinite(mat).sum())
        raise ValueError(f"Table is incomplete: {missing} (alpha,tau) cells are missing.")
    return alphas, taus, mat


def _configure_matplotlib(font_scale: float) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Sans-serif is typically clearer for dense heatmaps in papers.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.labelsize": 12.0 * float(font_scale),
            "axes.titlesize": 12.0 * float(font_scale),
            "xtick.labelsize": 10.0 * float(font_scale),
            "ytick.labelsize": 10.0 * float(font_scale),
        }
    )


def _plot_heatmap(
    out_path: Path,
    alphas: np.ndarray,
    taus: np.ndarray,
    mat: np.ndarray,
    title: Optional[str],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    annot: bool,
    annot_fmt: str,
    highlight_min: bool,
    figsize: Tuple[float, float],
    font_scale: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    _configure_matplotlib(font_scale=float(font_scale))

    vmin_f = float(np.min(mat)) if vmin is None else float(vmin)
    vmax_f = float(np.max(mat)) if vmax is None else float(vmax)
    if vmin_f == vmax_f:
        vmin_f -= 1e-12
        vmax_f += 1e-12

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True, dpi=int(dpi))
    im = ax.imshow(mat, cmap=str(cmap), vmin=vmin_f, vmax=vmax_f, aspect="auto")

    # Ticks (put grid at cell boundaries; ticks at centers).
    ax.set_xticks(np.arange(taus.size))
    ax.set_yticks(np.arange(alphas.size))
    ax.set_xticklabels([f"{t:g}" for t in taus.tolist()])
    ax.set_yticklabels([f"{a:g}" for a in alphas.tolist()])

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\alpha$")
    if title:
        ax.set_title(str(title))

    # Cell borders (subtle, paper-friendly).
    ax.set_xticks(np.arange(-0.5, taus.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, alphas.size, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MSE (lower is better)")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Optional numeric annotations.
    if annot:
        def rel_luminance(rgb: Tuple[float, float, float]) -> float:
            r, g, b = rgb
            return 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)

        ann_fs = 9.5 * float(font_scale)
        for i in range(alphas.size):
            for j in range(taus.size):
                val = float(mat[i, j])
                rgba = im.cmap(im.norm(val))
                lum = rel_luminance((float(rgba[0]), float(rgba[1]), float(rgba[2])))
                color = "#111111" if lum > 0.6 else "#FFFFFF"
                ax.text(
                    j,
                    i,
                    format(val, annot_fmt),
                    ha="center",
                    va="center",
                    fontsize=ann_fs,
                    color=color,
                )

    # Highlight the best (minimum) cell.
    if highlight_min:
        i_min, j_min = np.unravel_index(int(np.argmin(mat)), mat.shape)
        rect = patches.Rectangle(
            (j_min - 0.5, i_min - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="#D62728",
            linewidth=2.2,
        )
        ax.add_patch(rect)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot alpha/tau -> MSE heatmap (paper-style).")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="-",
        help="Input table path (tsv/csv). Use '-' to read stdin. Columns: alpha,tau,mse.",
    )
    parser.add_argument("--demo", action="store_true", help="Ignore --in and use the built-in demo table.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "plot" / "heatmaps" / "alpha_tau_mse_heatmap.pdf"),
        help="Output figure path (.pdf/.png).",
    )
    parser.add_argument("--title", type=str, default="Effect of $\\alpha$ and $\\tau$ on MSE")
    # Default to a colorblind-friendly, print-friendly palette and reverse so "lower is better" is brighter.
    parser.add_argument("--cmap", type=str, default="cividis_r")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--annot", action="store_true", help="Write numeric MSE in each cell.")
    parser.add_argument("--annot_fmt", type=str, default=".4f", help="Python format spec (default: .4f).")
    parser.add_argument("--no_highlight_min", action="store_true", help="Disable best-cell red box.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.4, 4.0))
    parser.add_argument("--font_scale", type=float, default=1.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--summary_json",
        type=str,
        default="",
        help="Optional path to write a summary json (best alpha/tau/mse and ranges).",
    )
    args = parser.parse_args()

    text = DEMO_TSV if args.demo else _read_text_input(args.in_path)
    alpha, tau, mse = _load_table(text)
    alphas, taus, mat = _make_matrix(alpha, tau, mse)

    out_path = Path(args.out)
    _plot_heatmap(
        out_path=out_path,
        alphas=alphas,
        taus=taus,
        mat=mat,
        title=args.title,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annot=bool(args.annot),
        annot_fmt=str(args.annot_fmt),
        highlight_min=(not bool(args.no_highlight_min)),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        font_scale=float(args.font_scale),
        dpi=int(args.dpi),
    )

    if args.summary_json:
        i_min, j_min = np.unravel_index(int(np.argmin(mat)), mat.shape)
        summary = {
            "best": {
                "alpha": float(alphas[i_min]),
                "tau": float(taus[j_min]),
                "mse": float(mat[i_min, j_min]),
            },
            "mse": {"min": float(np.min(mat)), "max": float(np.max(mat))},
            "alpha_values": [float(x) for x in alphas.tolist()],
            "tau_values": [float(x) for x in taus.tolist()],
        }
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
