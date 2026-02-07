# Rotational distribution calibration in DartQuant


In this directory, we provide the torch scripts for the experiments in DartQuant. 


## one-shot calibration data collection 

Currently, we only support **LLaMa** models. You can simply run the `get_train_data.py` to collect the calibration data. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--calib_dataset`: dataset used to obtain the calibration set
- `--r_path`: where to save the r1&2 calibration data
- `--r_list`: which rotation matrix calibration data to obtain
- `--nsamples`: number of samples
- `--seqlen`: sequence length
  
For example, to get the calibration data for Llama 2-7b, you can run the following command:

```bash
python get_train_data.py --model meta-llama/Llama-2-7b-hf --calib_dataset wikitext2 --nsamples 128 --seqlen 2048
```

## iterative rotation calibration


Currently, we only support **LLaMa** models. You can simply run the `r1_base_qr.py` to calibrate R1. The most important arguments are:


- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training
  
For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r1_base_qr.py --model meta-llama/Llama-2-7b-hf --ep 10 --bsz 64 --lr 1.5e-3
```

You can simply run the `r2_base_qr.py` to calibrate R2. The most important arguments are:


- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training

For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r2_base_qr.py --model meta-llama/Llama-2-7b-hf --ep 10 --bsz 64 --lr 1e-3
```

## stage-II Kronecker residual calibration

You can run `rkron_base_qr.py` to calibrate the layer-wise Kronecker residuals.
The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--r1_path`: path to the global R1 (R_g)
- `--target`: stage2 target (`r1`, `o_proj`, `down_proj`). For `r1`, the script learns two Kronecker rotations per layer: one for attention inputs (`q/k/v`) and one for MLP inputs (`up/gate`).
- `--rkron_n1`/`--rkron_n2`: Kronecker factor sizes (defaults from `get_decompose_dim`)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training

For example, to calibrate Kronecker residuals for Llama 2-7b, you can run:

```bash
python rkron_base_qr.py --model meta-llama/Llama-2-7b-hf --r1_path /path/to/r1.pt --ep 10 --bsz 64 --lr 1e-3
```

For `o_proj` you should also pass `--target o_proj --r2_path /path/to/r2.pt`.  
For `down_proj` you should prepare `r4` data via `get_train_data.py --r_list r4`.

Note: `rkron_base_qr.py` learns unconstrained (not necessarily orthogonal) Kronecker factors; inference fuses them using inverse-transpose.

Optional training regularizers (no STE):
- `--lambda_w`: add QSR loss on fused weights `W K^{-T}` (uses full weight matrices)
- `--lambda_cond`: add conditioning regularization on Kronecker factors
- `--inv_eps`: diagonal damping used for inverse/regularization
