import math
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform
from typing import Tuple


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def apply_kron_right(x, r1, r2):
    r1 = r1.to(device=x.device, dtype=x.dtype)
    r2 = r2.to(device=x.device, dtype=x.dtype)
    n1 = r1.shape[0]
    n2 = r2.shape[0]
    if x.shape[-1] != n1 * n2:
        raise ValueError(f'Invalid kron shape: x.shape[-1]={x.shape[-1]} but n1*n2={n1*n2}')
    orig_shape = x.shape
    x = x.reshape(-1, n1, n2)
    x = torch.einsum('ij,bjk->bik', r1.T, x)
    x = torch.einsum('bij,jk->bik', x, r2)
    return x.reshape(orig_shape)


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(round_ste(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.


def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.lac = False
        self.clip_factor_a_max = torch.nn.Parameter(torch.tensor(4.0), requires_grad=False)
        self.clip_factor_a_min = torch.nn.Parameter(torch.tensor(4.0), requires_grad=False)

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype

        if self.bits == 16:
            return x

        self.find_params(x)  # 首次量化参数计算

        if self.residual:
            if self.sym:
                tmp_q = sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = sym_quant_dequant(x - tmp_q, self.scale, self.maxq).to(x_dtype)
                return tmp_q + residual_q
            else:
                tmp_q = asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = asym_quant_dequant(x - tmp_q, self.scale, self.zero, self.maxq).to(x_dtype)
                return tmp_q + residual_q
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        else:
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits,
                  groupsize=-1,
                  sym=False,
                  clip_ratio=1.0,
                  residual=False,
                  lac=False,
                  lac_init=4.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'
        self.residual = residual
        prev_lac = getattr(self, 'lac', False)
        self.lac = lac
        if self.lac and not prev_lac:
            with torch.no_grad():
                self.clip_factor_a_max.fill_(float(lac_init))
                self.clip_factor_a_min.fill_(float(lac_init))
        self.clip_factor_a_max.requires_grad_(self.lac)
        self.clip_factor_a_min.requires_grad_(self.lac)

    def _get_clip_factors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.lac:
            one = x.new_tensor(1.0)
            return one, one
        # FlatQuant behavior: clip factors are learned in FP32 and applied via sigmoid.
        # Keep factors in FP32 for numerical stability (do not cast to x.dtype).
        max_factor = torch.sigmoid(self.clip_factor_a_max.float()).to(device=x.device)
        min_factor = torch.sigmoid(self.clip_factor_a_min.float()).to(device=x.device)
        return max_factor, min_factor

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize).float()

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True)
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True)
        # Ensure the quantization range includes 0 (matches token-wise path and FlatQuant).
        zero = torch.zeros_like(xmax)
        xmax = torch.maximum(xmax, zero)
        xmin = torch.minimum(xmin, zero)

        if self.lac:
            max_factor, min_factor = self._get_clip_factors(reshaped_x)
            xmax = xmax * max_factor
            xmin = xmin * min_factor
        else:
            xmax = xmax * self.clip_ratio
            xmin = xmin * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)    # QuaRot 源码,不要加会导致推理时间X10
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp)
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp)

        if self.lac:
            max_factor, min_factor = self._get_clip_factors(reshaped_x)
            xmax = xmax * max_factor
            xmin = xmin * min_factor
        else:
            xmax = xmax * self.clip_ratio
            xmin = xmin * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module: torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.register_buffer('kron_r1', torch.tensor(0))
        self._buffers['kron_r1'] = None
        self.register_buffer('kron_r2', torch.tensor(0))
        self._buffers['kron_r2'] = None
        self.kron_n1 = 0
        self.kron_n2 = 0
        self.online_kron = False
        # When training kron factors, using FP32 improves stability.
        self.kron_fp32 = False
        # Optional learnable per-channel scaling (SmoothQuant-style diagonal scale):
        # Conceptually inserts S between activation A and weight W:
        #   W' = W * S,  A' = A / S
        # Implemented efficiently as:
        #   A_q = quant(A / S) * S
        # so we do not need to materialize W * S during training/inference.
        self.channel_scale = False
        self.s_log_scale_clamp = 8.0
        self.s_log_scale = torch.nn.Parameter(
            torch.zeros((module.in_features,), dtype=torch.float32),
            requires_grad=False,
        )

    def enable_channel_scale(self, enabled: bool = True, *, init: float = 0.0) -> None:
        self.channel_scale = bool(enabled)
        with torch.no_grad():
            self.s_log_scale.fill_(float(init))
        self.s_log_scale.requires_grad_(self.channel_scale)

    def _get_channel_scale(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        log_s = self.s_log_scale.clamp(min=-self.s_log_scale_clamp, max=self.s_log_scale_clamp)
        scale = torch.exp(log_s).to(device=device, dtype=dtype)
        return scale

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_kron:
            r1 = getattr(self, "kron_r1_param", None)
            r2 = getattr(self, "kron_r2_param", None)
            if r1 is None:
                r1 = self.kron_r1
            if r2 is None:
                r2 = self.kron_r2
            if self.kron_fp32 and x_dtype != torch.float32:
                x = apply_kron_right(x.float(), r1.float(), r2.float()).to(x_dtype)
            else:
                x = apply_kron_right(x, r1, r2)
        elif self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1 / math.sqrt(init_shape[-1] // self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1,
                     init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        # Learnable per-channel scaling around activation quantization (applied only when quantized).
        scale_view = None
        if self.channel_scale and self.quantizer.bits < 16:
            scale = self._get_channel_scale(device=x.device, dtype=x_dtype)
            scale_view = scale.view(*([1] * (x.dim() - 1)), -1)
            x = x / scale_view

        if self.quantizer.bits < 16:  # Quantize, if needed
            # self.quantizer.find_params(x)  # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
            if scale_view is not None:
                x = x * scale_view

        # `torch.nn.Linear` requires input and weight dtype to match.
        # During kron/LAC training we may temporarily keep weights in FP32 for stability.
        w_dtype = self.module.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(w_dtype)
        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            # self.out_quantizer.find_params(x) # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        groupsize=-1, static_groups=False
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

        self.groupsize = groupsize
        self.static_groups = static_groups

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin.masked_fill(tmp, -1)
            xmax.masked_fill(tmp, +1)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1),
                                           zero1.unsqueeze(1), self.maxq)

                q = (q - x).abs().pow(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                best = torch.where(tmp, err, best)
                self.scale = torch.where(tmp, scale1, self.scale)
                self.zero = torch.where(tmp, zero1, self.zero)

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero,
                                      self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)


def find_qlayers(module, layers=[torch.nn.Linear,
                                 ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res



def disable_act_quant(module):
    bits_config = {}
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            bits_config[name] = m.quantizer.bits
            m.quantizer.bits = 16
            if m.out_quantizer.bits != 16:
                bits_config[name+'out'] = m.out_quantizer.bits
                m.out_quantizer.bits = 16

    return bits_config


def enable_act_quant(module, bits_config):
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.quantizer.bits = bits_config[name]
            if name+'out' in bits_config.keys():
                m.out_quantizer.bits = bits_config[name+'out']
