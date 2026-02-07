import math
import time
import tqdm
import torch
import torch.nn as nn
import logging
import functools

import utils
import quant_utils


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


import torch,torch.nn as nn,torch.nn.functional as F
import math,time,tqdm,gc

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

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
        mse=False, norm=2.4, grid=100, maxshrink=.8,pot=False
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)
        self.pot = pot

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
            # xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) # TODO 
            # self.scale = xmax / self.maxq
            self.scale = torch.maximum((xmin.clip(max=0))/(-self.maxq-1) ,xmax.clip(min=1e-6)/(self.maxq))
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = torch.maximum((xmin1.clip(max=0))/(-self.maxq-1) ,xmax1.clip(min=1e-6)/(self.maxq))
                    # scale1 = xmax1 / self.maxq
                    if self.pot:
                        scale1 = 2**(scale1.log2().ceil())
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
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
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if isinstance(module,tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class GPTQv2:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float().to(self.dev) * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())

        del self.fp_inp[0]

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, args=None
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        alpha = 0.25
        P = alpha * ((self.dXXT @ Hinv.T).triu(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


class FPInputsCache:
    """
    class for saving the full-precision output in each layer.
    """
    def __init__(self, sequential):
        self.fp_cache = {}
        # self.names = sequential[0]+sequential[1]+sequential[2]+sequential[3]
        self.names = sequential
        print(self.names)
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        assert len(inp.shape) == 3, f'inp.shape: {inp.shape}'
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name].append(inp.t().to(device='cpu'))

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


@torch.no_grad()
def gptqv2_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''

    print('-----GPTQv2 Quantization-----')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device="cpu", pin_memory=True
    )
    # inps = inps.cpu()
    # fp_inps = inps.clone()
    cache = {'i': 0, 'attention_mask': None}
    other_caches = dict()

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if kwargs.get("cache_position",None) is not None:
                other_caches['cache_position'] = kwargs['cache_position']
            if kwargs.get("position_embeddings",None) is not None:
                other_caches['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
        
    # class FPCatcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module

    #     def forward(self, inp, **kwargs):
    #         fp_inps[cache['i']] = inp
    #         cache['i'] += 1
    #         cache['attention_mask'] = kwargs['attention_mask']
    #         cache['position_ids'] = kwargs['position_ids']
    #         raise ValueError
    
    # collect fp inputs TO DO fp/quant separate collect
    # collect inputs
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    # model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps, pin_memory=True)

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module']
    ]
    first_module_name_list = [names[0] for names in sequential]
    fp_inputs_cache = FPInputsCache(first_module_name_list)
    fp_inps = inps.clone().pin_memory()
    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQv2 Quant.) Layers"):
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        # disable act quant
        bits_config = quant_utils.disable_act_quant(layer)
        first_module= {name: full[name] for name in first_module_name_list}
        fp_inputs_cache.add_hook(first_module)

        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,**other_caches)[0].cpu()
        fp_inputs_cache.clear_hook()

        # enable act quant
        quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                # print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.w_bits_down_proj is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj
                gptq[name] = GPTQv2(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                # gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            first_module_name = list(subset.keys())[0]
            gptq[first_module_name].fp_inp = fp_inputs_cache.fp_cache[first_module_name]
            handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,**other_caches)[0].cpu()
            handle.remove()

            # copy H and dXXT
            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False, args=args
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,**other_caches)[0].cpu()

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print('-----GPTQv2 Quantization Done-----\n')
    return quantizers