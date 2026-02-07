import logging
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

import model_utils
import quant_utils
import rotation_utils

import geoopt  # type: ignore


def _get_cali_dtype(model: torch.nn.Module, dev: torch.device, args) -> torch.dtype:
    """
    FlatQuant-style dtype policy for layer-wise calibration buffers:
    - CPU / non-CUDA: fp32
    - --deactive_amp: fp32
    - LLaMA on CUDA: fp16 (FlatQuant default)
    - otherwise: keep model dtype if fp16/bf16, else fp16
    """
    if dev.type != "cuda" or (not torch.cuda.is_available()):
        return torch.float32
    if bool(getattr(args, "deactive_amp", False)):
        return torch.float32
    if isinstance(model, model_utils.LLAMA_MODEL):
        return torch.float16
    dtype = next(model.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return torch.float16


def _get_layerwise_loss(args):
    """
    Loss for layer-wise calibration.
    - mse: mean-squared error on hidden states
    - kl: KL divergence between softmax distributions over hidden dim (teacher=fp, student=quant)
    """
    loss_type = str(getattr(args, "layerwise_loss", "mse") or "mse").lower()
    if loss_type not in ("mse", "kl"):
        logging.warning(f"Unknown --layerwise_loss={loss_type}; falling back to mse.")
        loss_type = "mse"

    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss()

        def _loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
            return loss_fn(y, y_hat)

        return _loss, "mse"

    temperature = float(getattr(args, "layerwise_kl_temp", 1.0) or 1.0)
    if temperature <= 0:
        logging.warning(f"Invalid --layerwise_kl_temp={temperature}; using 1.0.")
        temperature = 1.0
    t = float(temperature)

    def _loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # Flatten to 2D: (N, hidden)
        y_f = y.float().reshape(-1, y.shape[-1])
        y_hat_f = y_hat.float().reshape(-1, y_hat.shape[-1])
        target = F.softmax(y_f / t, dim=-1)
        log_q = F.log_softmax(y_hat_f / t, dim=-1)
        return F.kl_div(log_q, target, reduction="batchmean") * (t * t)

    return _loss, "kl"


def _get_traincast(model: torch.nn.Module, dev: torch.device, args):
    """
    FlatQuant-style AMP control:
    - default: autocast on CUDA using fp16 for LLaMA (FlatQuant default)
    - --deactive_amp: disable autocast (fp32)
    """
    from contextlib import nullcontext

    if dev.type != "cuda" or (not torch.cuda.is_available()):
        return nullcontext
    if bool(getattr(args, "deactive_amp", False)):
        return nullcontext
    dtype = _get_cali_dtype(model, dev, args)
    if dtype == torch.float32:
        return nullcontext
    return lambda: torch.amp.autocast(device_type="cuda", dtype=dtype)


def _get_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    t_max: int,
    eta_min: Optional[float] = None,
    args,
):
    if eta_min is None:
        eta_min = float(base_lr) * 1e-3
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=float(eta_min),
    )
    if not bool(getattr(args, "warmup", False)):
        return scheduler_main
    warmup_iters = min(16, max(1, t_max))
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_iters,
    )
    return torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])



def _collect_lac_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in model.modules():
        if hasattr(module, "clip_factor_a_max") and hasattr(module, "clip_factor_a_min"):
            if hasattr(module, "lac") and (not getattr(module, "lac", False)):
                continue
            max_param = getattr(module, "clip_factor_a_max")
            min_param = getattr(module, "clip_factor_a_min")
            if isinstance(max_param, torch.nn.Parameter):
                params.append(max_param)
            if isinstance(min_param, torch.nn.Parameter):
                params.append(min_param)
    # de-dup (shared params are unlikely, but be safe)
    params = list(dict.fromkeys(params))
    return params


def _collect_channel_scale_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, quant_utils.ActQuantWrapper):
            continue
        if not bool(getattr(module, "channel_scale", False)):
            continue
        p = getattr(module, "s_log_scale", None)
        if isinstance(p, torch.nn.Parameter):
            params.append(p)
    params = list(dict.fromkeys(params))
    return params


def _empty_cache(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_actquant_bits(module: torch.nn.Module, bits: int) -> List[tuple[torch.nn.Module, int]]:
    """
    Set all ActQuantizer.bits inside `module` to `bits`, returning previous bits for restore.
    """
    prev: List[tuple[torch.nn.Module, int]] = []
    for child in module.modules():
        if child.__class__.__name__ == "ActQuantizer" and hasattr(child, "bits"):
            prev.append((child, int(child.bits)))
            child.bits = bits
    return prev


def _restore_actquant_bits(prev: List[tuple[torch.nn.Module, int]]) -> None:
    for q, bits in prev:
        q.bits = bits


def lac_layerwise_ft(
    model: torch.nn.Module,
    trainloader: Iterable,
    dev: torch.device,
    args,
) -> None:
    """
    FlatQuant-style layer-wise calibration for LAC (+ optional per-channel scaling):
    - Capture hidden-states entering the first transformer layer
    - For each transformer layer i:
        1) Compute FP outputs with activation quantization disabled (bits=16)
        2) Optimize that layer's learnable clip/scale params to match FP outputs (MSE)
        3) Propagate FP outputs as next layer inputs
    """
    model.eval()
    if not isinstance(model, model_utils.LLAMA_MODEL):
        raise NotImplementedError("lac_layerwise_ft currently supports LLaMA models only.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    for _, param in model.named_parameters():
        param.requires_grad_(False)

    layers = model_utils.get_layers(model)
    nsamples = int(args.nsamples)
    seqlen = int(model.seqlen)
    hidden = int(model.config.hidden_size)
    cali_dtype = _get_cali_dtype(model, dev, args)

    for layer in layers:
        layer.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    inps = torch.zeros((nsamples, seqlen, hidden), dtype=cali_dtype, device=dev)
    cache: dict = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp[0]
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in trainloader:
            if cache["i"] >= nsamples:
                break
            try:
                sample = batch[0].to(dev)
                model(sample)
            except ValueError:
                pass

    attention_mask = cache.get("attention_mask", None)
    position_ids = cache.get("position_ids", None)

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    _empty_cache(dev)

    fp_inps = inps
    fp_outs = torch.zeros_like(fp_inps)
    loss_func, loss_name = _get_layerwise_loss(args)

    epochs = int(getattr(args, "lac_epochs", 1) or 1)
    cali_bsz = int(getattr(args, "lac_cali_bsz", 1) or 1)
    cali_bsz = max(1, min(cali_bsz, nsamples))
    log_interval = int(getattr(args, "lac_log_interval", 10) or 10)

    if attention_mask is not None and cali_bsz > 1:
        if attention_mask.dim() == 4:
            attention_mask_batch = attention_mask.repeat(cali_bsz, 1, 1, 1).float()
        elif attention_mask.dim() == 2:
            attention_mask_batch = attention_mask.repeat(cali_bsz, 1)
        else:
            attention_mask_batch = attention_mask
    else:
        attention_mask_batch = attention_mask

    for layer_idx in range(len(layers)):
        logging.info(f"========= LAC Layer {layer_idx} =========")
        layer = layers[layer_idx].to(dev)

        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()

        prev_bits = _set_actquant_bits(layer, 16)
        with torch.no_grad():
            for j in range(nsamples):
                fp_outs[j] = layer(
                    fp_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        _restore_actquant_bits(prev_bits)

        lac_params = _collect_lac_params(layer)
        scale_params = _collect_channel_scale_params(layer)
        if len(lac_params) == 0 and len(scale_params) == 0:
            logging.info("Skip: no learnable clip/scale params in this layer.")
            fp_inps, fp_outs = fp_outs, fp_inps
            layers[layer_idx] = layer.cpu()
            for name, param in layers[layer_idx].named_parameters():
                if name in dtype_dict:
                    param.data = param.data.to(dtype_dict[name])
            _empty_cache(dev)
            continue

        for p in lac_params:
            p.requires_grad_(True)

        for p in scale_params:
            p.requires_grad_(True)

        lac_lr = float(getattr(args, "lac_lr", 1e-2))
        lac_wd = float(getattr(args, "lac_wd", 0.0))
        s_lr_arg = getattr(args, "s_lr", None)
        s_lr = float(s_lr_arg) if s_lr_arg is not None else float(lac_lr) * 0.1
        s_wd = float(getattr(args, "s_wd", 0.0))

        traincast = _get_traincast(model, dev, args)
        trained_groups: list[dict] = []
        if len(lac_params):
            trained_groups.append({"params": lac_params, "lr": lac_lr, "weight_decay": lac_wd})
        if len(scale_params):
            trained_groups.append({"params": scale_params, "lr": s_lr, "weight_decay": s_wd})
        optimizer = torch.optim.AdamW(trained_groups)
        steps_per_epoch = nsamples // cali_bsz
        if steps_per_epoch <= 0:
            steps_per_epoch = 1
        base_lr_min = min(float(g["lr"]) for g in trained_groups)
        scheduler = _get_cosine_scheduler(
            optimizer,
            base_lr=base_lr_min,
            t_max=int(epochs) * int(steps_per_epoch),
            args=args,
        )

        for epoch in range(epochs):
            loss_acc = 0.0
            for j in range(steps_per_epoch):
                index = j * cali_bsz
                x = fp_inps[index:index + cali_bsz]
                y = fp_outs[index:index + cali_bsz]

                optimizer.zero_grad(set_to_none=True)
                with traincast():
                    quant_out = layer(
                        x,
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                    )[0]
                    loss_raw = loss_func(y, quant_out)
                loss_acc += float(loss_raw.detach().cpu())
                loss = loss_raw / loss_raw.detach().clamp(min=1e-12)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if log_interval > 0 and (epoch % log_interval == 0 or epoch == epochs - 1):
                logging.info(f"LAC layer {layer_idx} epoch {epoch + 1}/{epochs}: {loss_name}={loss_acc:.6f}")

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[layer_idx] = layer.cpu()
        for name, param in layers[layer_idx].named_parameters():
            if name in dtype_dict:
                param.data = param.data.to(dtype_dict[name])
        _empty_cache(dev)

    del inps, fp_inps, fp_outs
    _empty_cache(dev)
    model.config.use_cache = use_cache


def _collect_kron_wrappers(layer: torch.nn.Module) -> List[quant_utils.ActQuantWrapper]:
    wrappers: List[quant_utils.ActQuantWrapper] = []
    for module in layer.modules():
        if not isinstance(module, quant_utils.ActQuantWrapper):
            continue
        if not getattr(module, "online_kron", False):
            continue
        if getattr(module, "kron_r1", None) is None or getattr(module, "kron_r2", None) is None:
            continue
        wrappers.append(module)
    return wrappers


@torch.no_grad()
def _project_orthogonal_(mat: torch.Tensor) -> torch.Tensor:
    """
    Projects `mat` onto the nearest (in QR sense) orthogonal matrix.
    Returns the projected matrix (same shape/device/dtype).
    """
    q, r = torch.linalg.qr(mat.float())
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(mat.dtype)


def _enable_kron_params(
    wrappers: List[quant_utils.ActQuantWrapper],
    dev: torch.device,
    *,
    fp32: bool,
    orthogonal: bool,
) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    use_geoopt = bool(orthogonal) and (geoopt is not None)
    stiefel = geoopt.manifolds.Stiefel() if use_geoopt else None
    for w in wrappers:
        w.kron_fp32 = bool(fp32)
        if getattr(w, "kron_r1_param", None) is None:
            init_r1 = w.kron_r1.detach().to(device=dev, dtype=torch.float32)
            # if orthogonal:
            #     init_r1 = _project_orthogonal_(init_r1)
            if use_geoopt:
                w.kron_r1_param = geoopt.ManifoldParameter(init_r1, manifold=stiefel)  # type: ignore[arg-type]
            else:
                w.kron_r1_param = torch.nn.Parameter(init_r1)
        if getattr(w, "kron_r2_param", None) is None:
            init_r2 = w.kron_r2.detach().to(device=dev, dtype=torch.float32)
            # if orthogonal:
            #     init_r2 = _project_orthogonal_(init_r2)
            if use_geoopt:
                w.kron_r2_param = geoopt.ManifoldParameter(init_r2, manifold=stiefel)  # type: ignore[arg-type]
            else:
                w.kron_r2_param = torch.nn.Parameter(init_r2)
        w.kron_r1_param.requires_grad_(True)
        w.kron_r2_param.requires_grad_(True)
        params.extend([w.kron_r1_param, w.kron_r2_param])
    params = list(dict.fromkeys(params))
    return params


@torch.no_grad()
def _sync_kron_params_to_buffers(wrappers: List[quant_utils.ActQuantWrapper]) -> None:
    for w in wrappers:
        r1 = getattr(w, "kron_r1_param", None)
        r2 = getattr(w, "kron_r2_param", None)
        if r1 is None or r2 is None:
            continue
        # Keep both wrapper buffer and the underlying Linear buffers (if present) in sync.
        if getattr(w, "kron_r1", None) is not None:
            w.kron_r1.copy_(r1.detach().to(device=w.kron_r1.device, dtype=w.kron_r1.dtype))
        if getattr(w, "kron_r2", None) is not None:
            w.kron_r2.copy_(r2.detach().to(device=w.kron_r2.device, dtype=w.kron_r2.dtype))
        if hasattr(w, "module") and hasattr(w.module, "kron_r1") and w.module.kron_r1 is not None:
            w.module.kron_r1.copy_(r1.detach().to(device=w.module.kron_r1.device, dtype=w.module.kron_r1.dtype))
        if hasattr(w, "module") and hasattr(w.module, "kron_r2") and w.module.kron_r2 is not None:
            w.module.kron_r2.copy_(r2.detach().to(device=w.module.kron_r2.device, dtype=w.module.kron_r2.dtype))


def _disable_kron_params(wrappers: List[quant_utils.ActQuantWrapper]) -> None:
    for w in wrappers:
        # Removing params keeps state_dict stable (kron stays in existing buffers).
        if hasattr(w, "kron_r1_param"):
            w.register_parameter("kron_r1_param", None)
        if hasattr(w, "kron_r2_param"):
            w.register_parameter("kron_r2_param", None)
        w.kron_fp32 = False


@torch.no_grad()
def _prepare_kron_base_weights(
    wrappers: List[quant_utils.ActQuantWrapper],
) -> None:
    """
    Cache a per-wrapper base weight W_base so that the current fused Linear weight satisfies:
        W_fused = W_base @ K^{-T}
    where K = kron(r1, r2) is the wrapper's current kron.

    This lets us recompute W_fused from scratch each step (important for --no-kron_orthogonal),
    avoiding accumulation error from repeatedly applying incremental updates.
    """
    for w in wrappers:
        r1 = getattr(w, "kron_r1_param", None)
        r2 = getattr(w, "kron_r2_param", None)
        if r1 is None:
            r1 = w.kron_r1
        if r2 is None:
            r2 = w.kron_r2
        # W_base = W_fused @ K^T
        w._kron_base_weight = rotation_utils.apply_kron_right(w.module.weight.data, r1.T, r2.T).float()


@torch.no_grad()
def _recompute_fused_kron_weights_from_base(
    wrappers: List[quant_utils.ActQuantWrapper],
    *,
    inv_eps: float,
) -> None:
    """
    Refresh Linear weights to match the wrapper's current kron params:
        W_fused <- W_base @ K^{-T}
    This should be called before each forward when kron is unconstrained.
    """
    for w in wrappers:
        base = getattr(w, "_kron_base_weight", None)
        if base is None:
            continue
        r1 = getattr(w, "kron_r1_param", None)
        r2 = getattr(w, "kron_r2_param", None)
        if r1 is None or r2 is None:
            continue
        try:
            invt1 = rotation_utils.inv_transpose(r1.double(), eps=inv_eps).float()
            invt2 = rotation_utils.inv_transpose(r2.double(), eps=inv_eps).float()
        except RuntimeError as e:
            logging.warning(f"Kron inverse failed; skip fused-weight refresh this step: {e}")
            continue
        if (not torch.isfinite(invt1).all()) or (not torch.isfinite(invt2).all()):
            logging.warning("Kron inverse produced non-finite values; skip fused-weight refresh this step.")
            continue
        w.module.weight.data = rotation_utils.apply_kron_right(base, invt1, invt2).float()


@torch.no_grad()
def _cleanup_kron_base_weights(wrappers: List[quant_utils.ActQuantWrapper]) -> None:
    for w in wrappers:
        if hasattr(w, "_kron_base_weight"):
            delattr(w, "_kron_base_weight")


def kron_lac_layerwise_ft(
    model: torch.nn.Module,
    trainloader: Iterable,
    dev: torch.device,
    args,
) -> None:
    """
    Layer-wise calibration for (kron + LAC + optional per-channel scaling) before GPTQ:
    - Optimize each layer's learnable LAC clip factors (if enabled)
    - Optimize each wrapper's learnable per-channel scaling factors S (if enabled)
    - Optimize each ActQuantWrapper's kron factors inside that layer (if present)

    We keep the *FP* function invariant while training kron:
    - If `--kron_orthogonal`, enforce orthogonality via QR projection and apply incremental
      right-rotations to the underlying Linear weights.
    - If `--no-kron_orthogonal`, recompute fused weights from a cached base each step:
        W_fused = W_base @ K^{-T}
    """
    model.eval()
    if not isinstance(model, model_utils.LLAMA_MODEL):
        raise NotImplementedError("kron_lac_layerwise_ft currently supports LLaMA models only.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    for _, param in model.named_parameters():
        param.requires_grad_(False)

    layers = model_utils.get_layers(model)
    nsamples = int(args.nsamples)
    seqlen = int(model.seqlen)
    hidden = int(model.config.hidden_size)
    cali_dtype = _get_cali_dtype(model, dev, args)

    for layer in layers:
        layer.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    inps = torch.zeros((nsamples, seqlen, hidden), dtype=cali_dtype, device=dev)
    cache: dict = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp[0]
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in trainloader:
            if cache["i"] >= nsamples:
                break
            try:
                sample = batch[0].to(dev)
                model(sample)
            except ValueError:
                pass

    attention_mask = cache.get("attention_mask", None)
    position_ids = cache.get("position_ids", None)

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    _empty_cache(dev)

    fp_inps = inps
    fp_outs = torch.zeros_like(fp_inps)
    loss_func, loss_name = _get_layerwise_loss(args)
    mse_check = torch.nn.MSELoss()

    epochs = int(getattr(args, "kron_epochs", None) or getattr(args, "lac_epochs", 1) or 1)
    cali_bsz = int(getattr(args, "kron_cali_bsz", None) or getattr(args, "lac_cali_bsz", 1) or 1)
    cali_bsz = max(1, min(cali_bsz, nsamples))
    log_interval = int(getattr(args, "lac_log_interval", 10) or 10)
    kron_proj_every = int(getattr(args, "kron_proj_every", 1) or 1)
    kron_fp32 = bool(getattr(args, "kron_fp32", True))
    kron_orthogonal = bool(getattr(args, "kron_orthogonal", True))
    debug = bool(getattr(args, "kron_lac_debug", False))
    loss_norm = bool(getattr(args, "kron_lac_loss_norm", False))
    grad_clip = float(getattr(args, "kron_lac_grad_clip", 0.0) or 0.0)
    cos_lr = bool(getattr(args, "kron_lac_cos_lr", False))

    kron_inv_eps_arg = getattr(args, "kron_inv_eps", None)
    if kron_inv_eps_arg is None:
        kron_inv_eps = float(getattr(args, "rkron_inv_eps", 0.0) or 0.0)
        if (not kron_orthogonal) and kron_inv_eps == 0.0:
            kron_inv_eps = 1e-3
    else:
        kron_inv_eps = float(kron_inv_eps_arg)

    lac_lr = float(getattr(args, "lac_lr", 1e-2))
    lac_wd = float(getattr(args, "lac_wd", 0.0))
    kron_lr_arg = getattr(args, "kron_lr", None)
    if kron_lr_arg is None:
        # Unconstrained kron tends to become ill-conditioned quickly; use a conservative default lr.
        kron_lr = float(lac_lr if kron_orthogonal else min(lac_lr, 1e-5))
    else:
        kron_lr = float(kron_lr_arg)
    kron_wd = float(getattr(args, "kron_wd", 0.0))

    if attention_mask is not None and cali_bsz > 1:
        if attention_mask.dim() == 4:
            attention_mask_batch = attention_mask.repeat(cali_bsz, 1, 1, 1).float()
        elif attention_mask.dim() == 2:
            attention_mask_batch = attention_mask.repeat(cali_bsz, 1)
        else:
            attention_mask_batch = attention_mask
    else:
        attention_mask_batch = attention_mask

    for layer_idx in range(len(layers)):
        logging.info(f"========= Kron+LAC Layer {layer_idx} =========")
        layer = layers[layer_idx].to(dev)

        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()

        kron_wrappers = _collect_kron_wrappers(layer)
        # Match fp_outs precision path to training path.
        if len(kron_wrappers):
            with torch.no_grad():
                for w in kron_wrappers:
                    w.kron_fp32 = bool(kron_fp32)

        prev_bits = _set_actquant_bits(layer, 16)
        with torch.no_grad():
            for j in range(nsamples):
                fp_outs[j] = layer(
                    fp_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        _restore_actquant_bits(prev_bits)

        lac_params = _collect_lac_params(layer)
        scale_params = _collect_channel_scale_params(layer)
        kron_params = (
            _enable_kron_params(kron_wrappers, dev, fp32=kron_fp32, orthogonal=kron_orthogonal)
            if len(kron_wrappers)
            else []
        )
        use_geoopt = bool(kron_orthogonal) and (geoopt is not None) and len(kron_params) > 0
        if use_geoopt and layer_idx == 0:
            logging.info("Using geoopt manifold optimization for kron_orthogonal (RiemannianAdam).")
        if bool(kron_orthogonal) and (geoopt is None) and len(kron_params) > 0:
            logging.warning("geoopt is not available; falling back to Euclidean AdamW + QR projection for kron_orthogonal.")
        if len(kron_wrappers) and (not kron_orthogonal):
            _prepare_kron_base_weights(kron_wrappers)
            _recompute_fused_kron_weights_from_base(kron_wrappers, inv_eps=kron_inv_eps)
        if len(kron_wrappers):
            with torch.no_grad():
                prev_bits_check = _set_actquant_bits(layer, 16)
                check_out = layer(
                    fp_inps[0].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
                eq_mse = float(mse_check(fp_outs[0].unsqueeze(0), check_out).detach().cpu())
                logging.info(f"Kron equiv check layer {layer_idx}: fp_mse={eq_mse:.6e}")
                _restore_actquant_bits(prev_bits_check)

        if len(lac_params) == 0 and len(scale_params) == 0 and len(kron_params) == 0:
            logging.info("Skip: no learnable kron/LAC/scale params in this layer.")
            fp_inps, fp_outs = fp_outs, fp_inps
            layers[layer_idx] = layer.cpu()
            for name, param in layers[layer_idx].named_parameters():
                if name in dtype_dict:
                    param.data = param.data.to(dtype_dict[name])
            _empty_cache(dev)
            continue

        for p in lac_params:
            p.requires_grad_(True)
        for p in scale_params:
            p.requires_grad_(True)

        traincast = _get_traincast(model, dev, args)
        s_lr_arg = getattr(args, "s_lr", None)
        s_lr = float(s_lr_arg) if s_lr_arg is not None else float(lac_lr) * 0.1
        s_wd = float(getattr(args, "s_wd", 0.0))

        if debug:
            logging.info(
                f"Kron+LAC layer {layer_idx} params: lac_params={len(lac_params)} scale_params={len(scale_params)} "
                f"kron_wrappers={len(kron_wrappers)} kron_params={len(kron_params)} "
                f"(lac_lr={lac_lr:g}, s_lr={s_lr:g}, kron_lr={kron_lr:g}, kron_inv_eps={kron_inv_eps:g}, kron_orthogonal={kron_orthogonal}, geoopt={use_geoopt})"
            )
            lac_init = [p.detach().float().clone() for p in lac_params]
            scale_init = [p.detach().float().clone() for p in scale_params]
            kron_init = [(p.detach().float().clone()) for p in kron_params]
            saw_grad_log = False

        lac_optimizer = None
        kron_optimizer = None
        if len(lac_params) or len(scale_params):
            lac_groups: list[dict] = []
            if len(lac_params):
                lac_groups.append({"params": lac_params, "lr": lac_lr, "weight_decay": lac_wd})
            if len(scale_params):
                lac_groups.append({"params": scale_params, "lr": s_lr, "weight_decay": s_wd})
            lac_optimizer = torch.optim.AdamW(lac_groups)
        if len(kron_params):
            if use_geoopt:
                kron_optimizer = geoopt.optim.RiemannianAdam(  # type: ignore[union-attr]
                        kron_params,
                        lr=kron_lr,
                        weight_decay=kron_wd,
                        stabilize=10,
                    )
            else:
                kron_optimizer = torch.optim.AdamW(kron_params, lr=kron_lr, weight_decay=kron_wd)

        scheduler_lac = None
        scheduler_kron = None

        steps_per_epoch = nsamples // cali_bsz
        if steps_per_epoch <= 0:
            steps_per_epoch = 1
        global_step = 0
        if cos_lr:
            t_max = int(epochs) * int(steps_per_epoch)
            eta_min_arg = float(getattr(args, "kron_lac_eta_min", 0.0) or 0.0)
            if lac_optimizer is not None:
                lac_base_lr = min(float(g["lr"]) for g in lac_optimizer.param_groups)
                scheduler_lac = _get_cosine_scheduler(
                    lac_optimizer,
                    base_lr=lac_base_lr,
                    t_max=t_max,
                    eta_min=(eta_min_arg if eta_min_arg > 0 else None),
                    args=args,
                )
            if kron_optimizer is not None:
                scheduler_kron = _get_cosine_scheduler(
                    kron_optimizer,
                    base_lr=kron_lr,
                    t_max=t_max,
                    eta_min=(eta_min_arg if eta_min_arg > 0 else None),
                    args=args,
                )

        for epoch in range(epochs):
            loss_acc = 0.0
            for j in range(steps_per_epoch):
                index = j * cali_bsz
                x = fp_inps[index:index + cali_bsz]
                y = fp_outs[index:index + cali_bsz]

                if len(kron_wrappers) and (not kron_orthogonal):
                    # For unconstrained kron, always recompute fused weights from base for equivalence.
                    _recompute_fused_kron_weights_from_base(kron_wrappers, inv_eps=kron_inv_eps)
                elif len(kron_wrappers):
                    # Orthogonal case: incremental weight updates are stable and cheaper.
                    old_kron: list[tuple[quant_utils.ActQuantWrapper, torch.Tensor, torch.Tensor]] = []
                    with torch.no_grad():
                        for w in kron_wrappers:
                            old_kron.append(
                                (w, w.kron_r1_param.detach().clone(), w.kron_r2_param.detach().clone())
                            )

                if lac_optimizer is not None:
                    lac_optimizer.zero_grad(set_to_none=True)
                if kron_optimizer is not None:
                    kron_optimizer.zero_grad(set_to_none=True)
                with traincast():
                    quant_out = layer(
                        x,
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                    )[0]
                    loss_raw = loss_func(y, quant_out)
                loss_acc += float(loss_raw.detach().cpu())
                if loss_norm:
                    loss = loss_raw / loss_raw.detach().clamp(min=1e-12)
                else:
                    loss = loss_raw
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lac_params + scale_params + kron_params, max_norm=grad_clip)
                if debug and (not saw_grad_log):
                    lac_g = [p.grad for p in lac_params if p.grad is not None]
                    scale_g = [p.grad for p in scale_params if p.grad is not None]
                    kron_g = [p.grad for p in kron_params if p.grad is not None]
                    lac_gn = float(torch.stack([g.detach().float().norm() for g in lac_g]).mean().cpu()) if len(lac_g) else 0.0
                    scale_gn = float(torch.stack([g.detach().float().norm() for g in scale_g]).mean().cpu()) if len(scale_g) else 0.0
                    kron_gn = float(torch.stack([g.detach().float().norm() for g in kron_g]).mean().cpu()) if len(kron_g) else 0.0
                    logging.info(f"Kron+LAC layer {layer_idx} grad norms: lac={lac_gn:.3e} scale={scale_gn:.3e} kron={kron_gn:.3e}")
                    saw_grad_log = True
                if lac_optimizer is not None:
                    lac_optimizer.step()
                if kron_optimizer is not None:
                    kron_optimizer.step()
                if scheduler_lac is not None:
                    scheduler_lac.step()
                if scheduler_kron is not None:
                    scheduler_kron.step()

                global_step += 1
                if len(kron_wrappers) and kron_orthogonal:
                    with torch.no_grad():
                        do_project = (kron_proj_every > 0) and (global_step % kron_proj_every == 0)
                        for w, r1_old, r2_old in old_kron:
                            if do_project and (not use_geoopt):
                                w.kron_r1_param.copy_(_project_orthogonal_(w.kron_r1_param))
                                w.kron_r2_param.copy_(_project_orthogonal_(w.kron_r2_param))
                            r1_new = w.kron_r1_param
                            r2_new = w.kron_r2_param
                            # delta = kron(r_old)^T @ kron(r_new) = kron(r1_old^T r1_new, r2_old^T r2_new)
                            d1 = r1_old.T @ r1_new
                            d2 = r2_old.T @ r2_new
                            lin = w.module
                            lin.weight.data = rotation_utils.apply_kron_right(lin.weight.data, d1, d2)

                # if log_interval > 0 and (global_step % log_interval == 0):
                #     logging.info(f"Layer {layer_idx} step {global_step}: mse={mse_acc:.6f}")

            if log_interval > 0 and (epoch % log_interval == 0 or epoch == epochs - 1):
                logging.info(f"Kron+LAC layer {layer_idx} epoch {epoch + 1}/{epochs}: {loss_name}={loss_acc:.6f}")

        with torch.no_grad():
            if len(kron_wrappers) and (not kron_orthogonal):
                # Make sure stored weights correspond to the final kron params.
                _recompute_fused_kron_weights_from_base(kron_wrappers, inv_eps=kron_inv_eps)
            if len(kron_wrappers):
                prev_bits_check = _set_actquant_bits(layer, 16)
                check_out = layer(
                    fp_inps[0].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
                eq_mse = float(mse_check(fp_outs[0].unsqueeze(0), check_out).detach().cpu())
                logging.info(f"Kron equiv check layer {layer_idx} (end): fp_mse={eq_mse:.6e}")
                _restore_actquant_bits(prev_bits_check)
            _sync_kron_params_to_buffers(kron_wrappers)
            _cleanup_kron_base_weights(kron_wrappers)
        _disable_kron_params(kron_wrappers)
        if debug:
            lac_d = (
                float(torch.stack([(p.detach().float() - p0).abs().max() for p, p0 in zip(lac_params, lac_init)]).max().cpu())
                if len(lac_params)
                else 0.0
            )
            scale_d = (
                float(torch.stack([(p.detach().float() - p0).abs().max() for p, p0 in zip(scale_params, scale_init)]).max().cpu())
                if len(scale_params)
                else 0.0
            )
            kron_d = (
                float(torch.stack([(p.detach().float() - p0).abs().max() for p, p0 in zip(kron_params, kron_init)]).max().cpu())
                if len(kron_params)
                else 0.0
            )
            logging.info(f"Kron+LAC layer {layer_idx} param delta maxabs: lac={lac_d:.3e} scale={scale_d:.3e} kron={kron_d:.3e}")

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[layer_idx] = layer.cpu()
        for name, param in layers[layer_idx].named_parameters():
            if name in dtype_dict:
                param.data = param.data.to(dtype_dict[name])
        _empty_cache(dev)

    del inps, fp_inps, fp_outs
    _empty_cache(dev)
    model.config.use_cache = use_cache


__all__ = ["lac_layerwise_ft", "kron_lac_layerwise_ft"]
