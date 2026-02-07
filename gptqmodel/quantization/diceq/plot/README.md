# Plotting

## Activation distribution (origin / hadamard / SpinQuant / DICEQuant)

This repo stores the original per-layer activations in `.pt` files (e.g. `sample_<sid>_layer_<lid>_<module>.pt`).
`plot/plot_activation_distributions.py` will:

- load a (sampled) subset of these activations
- compute:
  - `origin`
  - `hadamard` (full Hadamard, scaled by `1/sqrt(d)`)
  - `spinquant` (right-multiply by `R1` from `spinquant_W16A4KV4_R.bin`)
  - `dicequant` (right-multiply by `R1` from the DICEQuant checkpoint)
- plot overlaid histograms (linear + log-y)

### Run

```bash
conda activate dartquant
python -m pip install -q matplotlib  # if missing

# default paths are already set to your current dataset/checkpoints
python plot/plot_activation_distributions.py

# 4 panels side-by-side (independent y-axis per method)
python plot/plot_activation_distributions.py --layout grid4

# Specific layer/module + sample count (grid4 uses all tokens by default)
python plot/plot_activation_distributions.py --layout grid4 --layers 1 --modules mlp_up_proj --nsamples 16

# Paper-quality export (bigger fonts + vector output)
python plot/plot_activation_distributions.py --layout grid4 --layers 1 --modules mlp_up_proj --nsamples 16 --out_format pdf --font_scale 1.4

# Optional knobs
# - disable smoothing: --smooth_sigma 0
# - show a suptitle: --grid4_title
# - log-y: --logy
# - higher raster quality: --dpi 600 (default), and adjust size: --figsize 18 4.2
# - change colors: --palette paper|okabe_ito|mono
# - show int4 quant grid + MSE: --quant_bits 4

# (recommended) plot per-module
python plot/plot_activation_distributions.py --modules self_attn_q_proj
python plot/plot_activation_distributions.py --modules mlp_up_proj
```

### Notes

- For speed, use sampling flags like `--max_files`, `--rows_per_file`, `--ndims`.
  - `--rows_per_file -1` is auto (grid4=all rows, overlay=32), `--rows_per_file 0` forces all token rows.
- Outputs are written to `plot/activation_distributions/` as:
  - `*.png` (figure)
  - `*.summary.json` (args + stats)
  - `*.hist.npz` (hist arrays)

## Per-layer quantile-curve compare (like min/percentiles lines)

`plot/plot_layer_quantile_compare.py` supports:

- `--layout grid4`: 4-panel quantile curves (origin/hadamard/spinquant/dicequant), y-axis not shared.
- `--layout compare`: origin vs each method (2 panels), shared y-axis.

Example: use 16 samples for `layer_1` `mlp_up_proj`:

```bash
conda activate dartquant
python plot/plot_layer_quantile_compare.py --layer 1 --module mlp_up_proj --num_samples 16 --layout grid4
```

Outputs:

- Figures: `plot/layer_quantile_compare/layer_<L>/<module>_origin_vs_<method>_ns<N>_tps<T>.png`
- Meta: `plot/layer_quantile_compare/layer_<L>/<module>_ns<N>_tps<T>.meta.json`

Common speed knobs:

- `--tokens_per_sample 256` (sample token rows per sample)
- `--random_samples --seed 0` (randomly pick sample ids)

## Alpha/Tau heatmap (MSE)

Plot a paper-style heatmap to visualize how `alpha` and `tau` affect `mse`:

```bash
conda activate dartquant

# quick demo (uses the built-in table from the paper note/chat)
python plot/plot_alpha_tau_mse_heatmap.py --demo --annot

# or read your own table from stdin (tsv/csv/whitespace; columns: alpha,tau,mse)
python plot/plot_alpha_tau_mse_heatmap.py --in - --annot --out plot/heatmaps/alpha_tau_mse.pdf <<'EOF'
alpha	tau	mse
5	0.005	0.098309
10	0.005	0.07896298
EOF
```
