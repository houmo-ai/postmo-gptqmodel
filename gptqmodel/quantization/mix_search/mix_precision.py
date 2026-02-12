# from .distribution import f, plot_fold_line
import contextlib
import copy
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.fx import Graph, GraphModule, Interpreter, Node
from torch.fx._compatibility import compatibility
from tqdm import tqdm

import xhquant
from xhquant.api import Config, Hook
from xhquant.backend.xh2a import HMFP
from xhquant.common.types import PrecisionMode
from xhquant.core import QBaseModule
from xhquant.lib import hsum
from xhquant.quantization.xh2a.qmodules import QConv2d, QConvTranspose2d, QLinear, QMoeBlock
from xhquant.utils.logger import get_root_logger
from xhquant.core.graph.quant_graph import enable_quanted_module
from ..common import QuantizerState


@contextlib.contextmanager
def device_guard(module: nn.Module, execution_device: torch.device):
    old_device = next(iter(module.parameters())).device
    module.to(execution_device)
    yield
    module.to(old_device)


class MixPrecisionProfilingHook(Hook):
    def __init__(self, graph_name: Optional[str] = None, metric_types=["hessian-mse"], smooth_alpha=0.9) -> None:
        super().__init__()
        self._graph_name = "" if graph_name is None else graph_name
        self.bit_candidates = [
            8,
        ]
        self.profile_results = dict()
        self.metric_types = metric_types
        self.smooth_alpha = smooth_alpha

    def before_node_run(self, graph_module: fx.GraphModule, graph: fx.Graph, node: fx.Node, args, kwargs) -> None:
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            if isinstance(module, nn.Linear):
                self.obtain_sensitivity_linear(node.target, module, args, kwargs)
            if isinstance(module, xhquant.nn.modules.normalized_modules.MatMul):
                self.obtain_sensitivity_matmul(node.target, module, args, kwargs)
            if isinstance(module, nn.Conv1d):
                self.obtain_sensitivity_conv1d(node.target, module, args, kwargs)
            if isinstance(module, nn.Conv2d):
                self.obtain_sensitivity_conv2d(node.target, module, args, kwargs)
        return args, kwargs

    def _append_metircs(
        self, bit, name, tensor, dim, gradient, gradient_transpose=False, squeeze=False, max_exp_boost=0
    ):
        if name.endswith("x1") or name.endswith("x2") or name.endswith("act"):
            hidden_bit = True
        else:
            hidden_bit = False

        use_hmfp = False
        if bit not in [8, 16]:
            use_hmfp = True

        if not tensor.is_cuda:
            use_hmfp = True

        if dim < 0:
            dim = dim + len(tensor.shape)

        if use_hmfp:
            hmfp = HMFP.from_tensor(
                tensor,
                mode="sefp",
                bit=bit,
                dim=dim,
                hidden_bit=hidden_bit,
                max_exp_boost=0,
                return_padded=True,
                assemble=False,
                inf_nan_replace=[0, 65504],
                hmfp_use_cuda=False,
            )
            if isinstance(hmfp, HMFP):
                quant = hmfp.to(torch.float16, rdn=False)
                if len(quant.shape) > len(tensor.shape):
                    quant = quant.flatten(dim, dim + 1)
                    assert len(quant.shape) == len(tensor.shape)
                if dim == 0:
                    quant = quant[: tensor.shape[dim]]
                elif dim == 1:
                    quant = quant[:, : tensor.shape[dim]]
                elif dim == 2:
                    quant = quant[:, :, : tensor.shape[dim]]
                elif dim == 3:
                    quant = quant[:, :, :, : tensor.shape[dim]]
                else:
                    raise RuntimeError("not support dim {}".format(dim))
            else:
                quant = hmfp
            hmfp = None
        else:
            return_padded = False
            quant = hsum.to_share_exp_fp16(
                tensor,
                dim,
                64,
                5,
                bit - 1,
                rounding="rne",
                keep_bit_for_exp=hidden_bit,
                return_padded=return_padded,
                assemble=True,
                exp_cuda_compatiable=True,
                max_exp_boost=max_exp_boost,
                inf_nan_replace=None,
                fix_all_subnorm=True,
            )[0]

        if squeeze:
            quant = quant.flatten(start_dim=dim - 1, end_dim=dim)
            tensor = tensor.flatten(start_dim=dim - 1, end_dim=dim)
            if gradient is not None:
                if gradient_transpose:
                    gradient = gradient.flatten(start_dim=-2, end_dim=-1)
                else:
                    gradient = gradient.flatten(start_dim=0, end_dim=1)

        # name = f"{name}.{bit}"
        if name not in self.profile_results:
            self.profile_results[name] = dict()

        if "l1" in self.metric_types:
            delta = tensor - quant
            delta = delta.float()
            sen = delta.abs().sum() / tensor.numel()
            sen = sen.item()
            if "l1" in self.profile_results[name]:
                last = self.profile_results[name]["l1"]
                sen = self.smooth_alpha * sen + (1 - self.smooth_alpha) * last
            delta = None
            self.profile_results[name]["l1"] = sen

        if "mse" in self.metric_types:
            delta = tensor - quant
            delta = delta.float()
            sen = (delta * delta).sum() / tensor.numel()
            sen = sen.item()
            if "mse" in self.profile_results[name]:
                last = self.profile_results[name]["mse"]
                sen = self.smooth_alpha * sen + (1 - self.smooth_alpha) * last
            delta = None
            self.profile_results[name]["mse"] = sen

        if "hessian-mse" in self.metric_types or "hessian_mse" in self.metric_types:
            hessian = pow(gradient.float(), 2)

            delta = tensor - quant
            delta = delta.float()
            if gradient_transpose:
                sen = torch.matmul(torch.matmul(hessian, delta), delta.transpose(-2, -1))
            else:
                sen = torch.matmul(delta.transpose(-1, -2), torch.matmul(delta, hessian))
            hessian = None
            delta = None

            sen = sen.sum() / sen.numel()
            sen = sen.item()
            local_max_exp_boost = max_exp_boost
            if "hmse" in self.profile_results[name]:
                last = self.profile_results[name]["hmse"]
                if "hmse-max_exp_boost" in self.profile_results[name]:
                    local_max_exp_boost = self.profile_results[name]["hmse-max_exp_boost"]

                if self.smooth_alpha >= 1.0:
                    if sen < last:
                        local_max_exp_boost = max_exp_boost
                    else:
                        sen = last
                else:
                    sen = self.smooth_alpha * sen + (1 - self.smooth_alpha) * last

            self.profile_results[name]["hmse"] = abs(sen)
            self.profile_results[name]["hmse-max_exp_boost"] = local_max_exp_boost

        if "cosine-similarity" in self.metric_types or "cosine_similarity" in self.metric_types:
            sen = torch.cosine_similarity(tensor, quant, dim=dim)
            sen = sen.sum() / tensor.numel()
            sen = sen.item()
            if "cos_sim" in self.profile_results[name]:
                last = self.profile_results[name]["cos_sim"]
                sen = self.smooth_alpha * sen + (1 - self.smooth_alpha) * last
            self.profile_results[name]["cos_sim"] = sen

        if "kld" in self.metric_types:
            sen = torch.nn.functional.kl_div(quant.log(), tensor)
            sen = sen.item()
            if "kld" in self.profile_results[name]:
                last = self.profile_results[name]["kld"]
                sen = self.smooth_alpha * sen + (1 - self.smooth_alpha) * last
            self.profile_results[name]["kld"] = sen

    def append_metircs(self, name, tensor, dim, gradient, gradient_transpose=False, squeeze=False, max_exp_boost=0):
        if not isinstance(tensor, torch.Tensor) and hasattr(tensor, "_data"):
            tensor = tensor._data
        if not isinstance(gradient, torch.Tensor) and hasattr(gradient, "_data"):
            gradient = gradient._data

        # for bit in self.bit_candidates:
        #     self._append_metircs(bit, name, tensor, dim, gradient, gradient_transpose, squeeze)
        bit = 8
        try:
            self._append_metircs(
                bit, name, tensor, dim, gradient, gradient_transpose, squeeze, max_exp_boost=max_exp_boost
            )
        except:
            self._append_metircs(
                bit, name, tensor.cpu(), dim, gradient.cpu(), gradient_transpose, squeeze, max_exp_boost=max_exp_boost
            )

    def obtain_sensitivity_linear(self, name, module, args, kwargs):
        act = args[0]  #  B, R, Cin or R, Cin
        assert len(module.weight.shape) == 2, f"module.weight.shape: {module.weight.shape}"
        weight = module.weight.t()

        if not isinstance(act, torch.Tensor) and hasattr(act, "_data"):
            act = act._data
        if not isinstance(weight, torch.Tensor) and hasattr(weight, "_data"):
            weight = weight._data

        self.append_metircs(f"{self._graph_name}.{name}.act", act, dim=-1, gradient=weight, gradient_transpose=False)

        for max_exp_boost in [0, -1, -2, -3, -4]:
            self.append_metircs(
                f"{self._graph_name}.{name}.weight", weight, dim=0, gradient=act, gradient_transpose=True
            )

        name_ = f"{self._graph_name}.{name}.act"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = act.numel()

        name_ = f"{self._graph_name}.{name}.weight"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = weight.numel()

    def obtain_sensitivity_matmul(self, name, module, args, kwargs):
        x1 = args[0]
        x2 = args[1]

        if not isinstance(x1, torch.Tensor) and hasattr(x1, "_data"):
            x1 = x1._data
        if not isinstance(x2, torch.Tensor) and hasattr(x2, "_data"):
            x2 = x2._data

        self.append_metircs(f"{self._graph_name}.{name}.x1", x1, dim=-1, gradient=x2, gradient_transpose=False)
        self.append_metircs(f"{self._graph_name}.{name}.x2", x2, dim=-2, gradient=x1, gradient_transpose=True)

        name_ = f"{self._graph_name}.{name}.x1"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = x1.numel()

        name_ = f"{self._graph_name}.{name}.x2"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = x2.numel()

    def obtain_sensitivity_conv1d(self, name, module, args, kwargs):
        raise NotImplementedError("conv1d")

    def obtain_sensitivity_conv2d(self, name, module, args, kwargs):
        act = args[0]  #  B, R, Cin or R, Cin

        if not isinstance(act, torch.Tensor) and hasattr(act, "_data"):
            act = act._data
        if not isinstance(module.weight, torch.Tensor) and hasattr(module.weight, "_data"):
            weight = weight._data
        else:
            weight = module.weight

        name_ = f"{self._graph_name}.{name}.act"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = act.numel()

        name_ = f"{self._graph_name}.{name}.weight"
        if name_ not in self.profile_results:
            self.profile_results[name_] = dict()
        self.profile_results[name_]["numel"] = weight.numel()

        B, C, _, _ = act.shape
        unfold = F.unfold(act, module.kernel_size, module.dilation, module.padding, module.stride)
        unfold = unfold.transpose(1, 2)
        R = unfold.shape[1]
        unfold = unfold.reshape(B, R, C, -1)
        unfold = unfold.transpose(2, 3)

        Co, Ci, Kh, Kw = weight.shape
        weight = weight.reshape(Co, Ci, -1).transpose(1, 2)  # Co KhKw Ci
        weight = weight.reshape(Co, -1).transpose(0, 1).reshape(-1, Ci, Co)

        self.append_metircs(
            f"{self._graph_name}.{name}.act", unfold, dim=-1, gradient=weight, gradient_transpose=False, squeeze=True
        )
        self.append_metircs(
            f"{self._graph_name}.{name}.weight", weight, dim=1, gradient=unfold, gradient_transpose=True, squeeze=True
        )

    def topk(self, k=10, metirc="hmse", filters=["."]):
        logger = get_root_logger()
        names = []
        values = []
        elements = []
        for name, v in self.profile_results.items():
            meet_filter = False
            for f in filters:
                if f in name:
                    meet_filter = True
                    break
            if not meet_filter:
                continue

            if metirc in v:
                values.append(v[metirc])
                names.append(name)
                if "numel" in v:
                    elements.append(v["numel"])

        values = np.array(values)
        values = torch.from_numpy(values)

        indice = []
        if k < 0:  # indicates percentage
            k = int(len(values) * -1.0 * k / 100)

        if k > len(values):
            k = len(values)

        logger.info("choosing Tok-{} in {}".format(k, len(values)))
        if len(values) != 0:
            _, indice = values.topk(k=k)

        if len(elements) != len(values):
            logger.info("missing numel")

        total_count = sum(elements)
        chosen_count = 0
        names_ = []
        for i in indice:
            target = names[i]
            target = ".".join(target.split(".")[1:-1])
            names_.append(target)
            chosen_count += elements[i]

        return names_, total_count, chosen_count

    def load(self, filename="profile_results.xlsx"):
        filename = filename.lower()
        if filename.endswith("csv"):
            df = pd.read_csv(filename)
        else:
            df = pd.read_excel(filename)
        df = df.transpose()

        self.profile_results = dict()
        for i in df:
            item = df[i]
            name = item["Unnamed: 0"]
            item.pop("Unnamed: 0")
            for metric, val in item.items():
                if name not in self.profile_results:
                    self.profile_results[name] = dict()
                self.profile_results[name][metric] = val

    def save(self, filename="profile_results.xlsx"):
        df = pd.DataFrame(self.profile_results)

        filename = filename.lower()
        if filename.endswith("csv"):
            df.transpose().to_csv(filename)
        else:
            df.transpose().to_excel(filename)


class MixPrecisionSearch(Interpreter):
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        module: torch.nn.Module,
        cfg: Dict = None,
        garbage_collect_values: bool = True,
        graph: Optional[Graph] = None,
    ):
        """
        Configures sensitivity analysis parameters for model quantization.

        Args:
            metric (str, optional): Strategy for sensitivity calculation.
                Supported values:
                    "kl" - Kullback-Leibler divergence between outputs of
                           floating-point and quantized models (lower value
                           indicates lower sensitivity).
                    "sqnr" - Signal-to-Quantization-Noise Ratio.
                Defaults to "kl".

            policy (str, optional): Method for selecting globally sensitive layers.
                Supported values:
                    "topk" - Select top K% layers by sensitivity (requires setting
                             a float value, e.g., 0.2 for top 20%).
                    "threshold" - Select layers exceeding specified error threshold.
                Defaults to "topk".

            key_name (str, optional): Name of the model output used in sensitivity calculation.
                Critical for large language models (LLMs) where outputs may be a dict containing logits or loss.
                Notes:
                    1. Caller MUST preprocess outputs when using KL divergence
                       (e.g., apply log-softmax and remove padding tokens).
                    2. Ignored if model output is a single tensor.
                Defaults to "logits".
        """
        super().__init__(module, garbage_collect_values, graph)
        self.config: Dict[str, Any] = {
            "weight_bits": [4, 8],
            "act_bits": [8],
            "policy": "topk",
            "threshold": 1e-6,
            "topk": 0.1,
            "metric": "kl",
            "key_name": "logits",
            "auto_quant_after_search": True,
            "default_w8a16": False,
            "default_w8a8": False,
            "task": "llm",
            "search_candidates": ["XH2aQuantQLinear","XH2aQuantQConv2d","XH2aQuantQConvTranspose2d","XH2aQuantQMoeBlock"],
        }
        if cfg is not None:
            self.config.update(cfg)
        self.candidate_ops: Optional[Set] = None


    @torch.no_grad()
    def load_search_result(self, f):
        import os
        if isinstance(f,str) and os.path.exists(f):
            logger = get_root_logger()
            logger.info(f"Loading search result from {f}")
            res = torch.load(f, weights_only=False)
        elif isinstance(f,dict):
            res = f
        else:
            raise ValueError(f"Invalid search result {f}")
        name2node = self._get_ops()
        for name, settings in res.items():
            assert name in name2node
            module = self.module.get_submodule(name2node[name].target)
            module.i_cfg = settings["i_cfg"]
            module.w_cfg = settings["w_cfg"]
            if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                module.clear_quant_state()
                module.fire_quant()
                with device_guard(module, "cuda"):
                    module.weight_static_quant()
                torch.cuda.empty_cache()
                module.fixed_quant()

    def save_search_result(self, f):
        result = self.get_search_result()
        if len(result) > 0:
            torch.save(result, f)

    def get_search_result(self):
        if self.candidate_ops is None:
            raise ValueError("You should call search before calling get_search_result().")
        result = {}
        for op in self.candidate_ops:
            module = self.module.get_submodule(op.target)
            if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                assert op.target not in result
                result[op.target] = {"i_cfg": module.i_cfg, "w_cfg": module.w_cfg}
        return result

    def _get_ops(self):
        name2node = {}
        for node in self.graph.nodes:
            if node.op == "call_module":
                module = self.module.get_submodule(node.target)
                if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                    name2node[node.target] = node
        return name2node

    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args,
        initial_env: Optional[dict[Node, Any]] = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)

        for node in self.graph.nodes:
            if node in self.env:
                continue
            try:
                if self.candidate_ops is None:
                    self.candidate_ops = set()
                if node.op == "call_module":
                    module = self.module.get_submodule(str(node.target))
                    # if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                    if module._get_name() in self.config["search_candidates"]:
                        self.candidate_ops.add(node)
                self.env[node] = self.run_node(node)
            except Exception as e:
                if self.extra_traceback:
                    msg = f"While executing {node.format_node()}"
                    msg = f"{e.args[0]}\n\n{msg}" if e.args else str(msg)
                    if (
                        isinstance(self.module, GraphModule)
                        and self.module.graph is not None
                        and isinstance(self.module.graph, fx.Graph)
                    ):
                        msg += f"\nGraphModule: {self.module.print_readable(print_output=False, include_stride=True)}\n"
                    msg += f"\nOriginal traceback:\n{node.stack_trace}"
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise e

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                self.env.clear()
                torch.cuda.empty_cache()
                return self.graph.process_outputs(output_val) if enable_io_processing else output_val

    def compute_loss(self, output, label=None):
        if self.config["task"] == "llm":
            if isinstance(output, dict):
                if self.config["key_name"] in output:
                    return output[self.config["key_name"]]
                else:
                    logits = None
                    for val in output.values():
                        logits = val
                        break
            if label is None:
                return output
            logits = output
            assert logits.ndim == 3
            # bs, seq_len, vocabulary_size
            bs = logits.shape[0]
            assert bs == 1, "we assume batch_size = 1"
            logits = logits.float()
            shift_labels = label[..., 1:].contiguous()
            vocabulary_size = logits.shape[-1]
            logits = logits[0, :-1].view(-1, vocabulary_size)
            shift_labels = shift_labels.view(-1).long()
            loss = nn.functional.cross_entropy(
                logits[:shift_labels.shape[-1]], shift_labels.to(logits.device), ignore_index=-100, reduction="sum"
            )
            return loss
        elif self.config["task"] == "cv_cls":
            loss = 0
            if isinstance(output, torch.Tensor):
                loss = nn.functional.cross_entropy(output, label[0])
            elif isinstance(output, dict):
                keys = list(output.keys())
                for key in keys:
                    loss += nn.functional.cross_entropy(output[key].float(), label[key].float())
            elif isinstance(output, (list, tuple)):
                for o, l in zip(output, label):
                    loss += nn.functional.cross_entropy(o.float(), l.float())
                return loss
            return loss
        else:
            if isinstance(output, torch.Tensor):
                loss = nn.functional.mse_loss(output.float(), label.float())
                return loss
            elif isinstance(output, dict):
                keys = list(output.keys())
                loss = 0
                for key in keys:
                    loss += nn.functional.mse_loss(output[key].float(), label[key].float())
                return loss
            elif isinstance(output, (list, tuple)):
                loss = 0
                for o, l in zip(output, label):
                    loss += nn.functional.mse_loss(o.float(), l.float())
                return loss
            else:
                raise ValueError(f"Unsupported output type: {type(output)}")
        return None

    def update_cfg(self, cfg):
        def apply_fn(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(cfg)

        self.module.apply(apply_fn)

    def set_input_sequence_length(self, input_sequence_length: int, wrap_cfg: None, num_logits_to_keep: int):
        # self.update_cfg(Config(dict(input_sequence_length = input_sequence_length)))
        wrap_cfg.input_sequence_length = input_sequence_length
        wrap_cfg.num_logits_to_keep = num_logits_to_keep
        self.update_cfg(wrap_cfg)

    @torch.no_grad()
    def search(
        self,
        data_loader,
        origin_mode,
        labels=None,
        wrap_cfg=None,
        initial_env: Optional[Dict[Node, Any]] = None,
        enable_io_processing: bool = True,
    ):
        op2sens = {}
        assert type(data_loader) in (list, tuple)
        assert len(data_loader) > 0
        # search_space = len(data_loader) * len(self.candidate_ops) * len(self.config["act_bits"]) * len(self.config["weight_bits"])
        search_space = 0
        logger = get_root_logger()
        pbar = tqdm(
            total=1,
            desc="search mix precision",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Remaining: {remaining}",
            dynamic_ncols=True,
        )
        for bi, data in enumerate(data_loader):
            self.module.disable_quant()  # get baseline result.
            # step 1, run under FP mode to get baseline.
            if isinstance(data, torch.Tensor):
                data = [data]
            if wrap_cfg is not None:
                self.set_input_sequence_length(data[0].shape[1], wrap_cfg, 0)

            output_fp = self.run(*data)
            if self.config["key_name"] == "loss":
                if labels is not None and len(labels) == len(data_loader):
                    label = labels[bi]
                else:
                    label = None
                res = self.compute_loss(output_fp, label)
                if res is not None:  # When we implmented self.config["task"]
                    output_fp = res
                else:
                    # we return to default key_name
                    self.config["key_name"] = "logits"
            if search_space == 0:
                search_space = (
                    len(data_loader)
                    * len(self.candidate_ops)
                    * len(self.config["act_bits"])
                    * len(self.config["weight_bits"])
                )
                pbar.total = search_space
                pbar.refresh()
                pbar.n = 0
            for op in self.candidate_ops:
                module: nn.Module = self.module.get_submodule(op.target)
                if not isinstance(module, QBaseModule):
                    continue
                qmodule: QBaseModule = module

                if isinstance(qmodule, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                    old_wfg = copy.deepcopy(qmodule.w_cfg)
                    old_icfg = copy.deepcopy(qmodule.i_cfg)
                    qmodule._is_quant_fixed = False
                    old_mode = qmodule.precision_mode
                    qmodule.precision_mode = PrecisionMode.FAST
                    with enable_quanted_module(qmodule):
                        for w_bit_width in self.config["weight_bits"]:
                            # qmodule.quant_state = QuantizerState.Quantizing
                            qmodule.w_cfg.qspec.man_bit = w_bit_width
                            qmodule.clear_quant_state()
                            for a_bit_width in self.config["act_bits"]:
                                module.i_cfg.qspec.man_bit = a_bit_width
                                out_qt = self.run(*data)
                                if self.config["key_name"] == "loss":
                                    if labels is not None and len(labels) == len(data_loader):
                                        label = labels[bi]
                                    else:
                                        label = None
                                    out_qt = self.compute_loss(out_qt, label)
                                    assert out_qt is not None
                                pbar.update(1)
                                sense = self.calculate_sensitivity(output_fp, out_qt)
                                if op not in op2sens:
                                    op2sens[op] = {
                                        "w4a4": None,
                                        "w4a5": None,
                                        "w4a8": None,
                                        "w4a16": None,
                                        "w5a5": None,
                                        "w5a8": None,
                                        "w5a16": None,
                                        "w6a5": None,
                                        "w6a8": None,
                                        "w6a16": None,
                                        "w8a8": None,
                                        "w8a16": None,
                                        "w16a8": None,
                                        "w16a16": None,
                                    }
                                key = "w%da%d" % (w_bit_width, a_bit_width)
                                if op2sens[op][key] is None:
                                    op2sens[op][key] = sense
                                else:
                                    op2sens[op][key] += sense
                    qmodule.w_cfg = old_wfg
                    qmodule.i_cfg = old_icfg
                    qmodule.precision_mode = old_mode
                # qmodule.quant_state = QuantizerState.NONE
        pbar.close()
        self.module.enable_quant(origin_mode)
        # Now comes to decide which op is less sensitive.
        sense_list: List[List[Any]] = []
        max_w_bit = 0
        max_a_bit = 0
        for node, val in op2sens.items():
            for wxay, sens in val.items():
                if sens is None:
                    continue
                w_bit = int(wxay.split("a")[0].split("w")[-1])
                a_bit = int(wxay.split("a")[-1])
                max_w_bit = max(max_w_bit, w_bit)
                max_a_bit = max(max_a_bit, a_bit)
                sense_list.append([sens / len(data_loader), node, w_bit, a_bit])
        # in descending order.
        sense_list.sort(key=lambda e: e[0], reverse=True)
        statics = self.adjust_bitwidth(sense_list, max_w_bit, max_a_bit)
        total_cnt = 0
        for key, s in statics.items():
            total_cnt += len(s)
        if wrap_cfg is not None:
            self.set_input_sequence_length(256, wrap_cfg, 1)

        headers = ["WxAy", "cnt", "ratio"]
        data_rows = []
        for key, s in statics.items():
            if len(s) == 0:
                continue
            data_rows.append([key, len(s), len(s) / total_cnt])
        # logger.info("statics:\n WxAy\tcnt\tratio\n{}".format("\n".join(str_infos)))

        table = PrettyTable(headers, float_format=".4f")
        table.add_rows(data_rows)
        logger.info("statics:\n{}".format(table))

    def calculate_sensitivity(
        self,
        fp_out,
        qt_out,
    ):
        """
        To ensure cross-scenario applicability of the interface, the following calculations assume the underlying call scenario (targeting LLMs or CV models) has processed its output.
        """
        if isinstance(fp_out, dict):
            if self.config["key_name"] in fp_out:
                key = self.config["key_name"]
            else:
                key = list(fp_out.keys())[0]
            fp_out = fp_out[key]
        if isinstance(qt_out, dict):
            if self.config["key_name"] in qt_out:
                key = self.config["key_name"]
            else:
                key = list(qt_out.keys())[0]
            qt_out = qt_out[key]
        assert isinstance(fp_out, torch.Tensor)
        assert isinstance(qt_out, torch.Tensor)
        if self.config["metric"] == "kl":
            sensitive = F.kl_div(
                F.log_softmax(qt_out, dim=-1), F.log_softmax(fp_out, dim=-1), reduction="batchmean", log_target=True
            )
            return sensitive.item()
        elif self.config["metric"] == "sqnr":
            # noise / signal
            sensitive = torch.mean((fp_out - qt_out) ** 2) / (torch.mean(fp_out**2) + 1e-6)
        elif self.config["metric"] == "l1":
            sensitive = torch.sum(torch.abs(fp_out - qt_out))
        else:
            raise NotImplementedError
        return sensitive.item()

    def adjust_bitwidth(self, sense_list, max_w_bit, max_a_bit):
        """
        sense list has the following format.
        (sensitive_metric, node, w_bit, act_bit)
        """
        statics = {
            "w4a4": set([]),
            "w4a5": set([]),
            "w4a8": set([]),
            "w4a16": set([]),
            "w5a5": set([]),
            "w5a8": set([]),
            "w5a16": set([]),
            "w6a5": set([]),
            "w6a8": set([]),
            "w6a16": set([]),
            "w8a8": set([]),
            "w8a16": set([]),
            "w16a8": set([]),
            "w16a16": set([]),
        }
        if self.config["policy"] in ("threshold", "topk"):
            I = 0
            if self.config["policy"] == "threshold":
                for i, v in enumerate(sense_list):
                    if v[0] <= self.config["threshold"]:
                        I = i
                        break
            else:
                I = int(len(sense_list) * self.config["topk"])
            # I = int(min(max(1, I), len(sense_list) * 0.15))
            high_fp_ops = {}
            low_fp_ops = {}
            for i, blob in enumerate(sense_list):
                if i < I:
                    if blob[1] not in high_fp_ops:
                        high_fp_ops[blob[1]] = []
                    high_fp_ops[blob[1]].append([i, blob[2], blob[3]])
                else:
                    if blob[1] not in low_fp_ops:
                        low_fp_ops[blob[1]] = []
                    low_fp_ops[blob[1]].append([i, blob[2], blob[3]])
            visit = set([])
            num_high_fp_ops = int(len(self.candidate_ops) * self.config["topk"]) + 1
            for i in range(I):
                node = sense_list[i][1]
                if len(visit) >= num_high_fp_ops:
                    break
                # This module has already been set.
                if node.target in visit:
                    continue
                # all sensitive
                module = self.module.get_submodule(node.target)
                assert node.op == "call_module"
                if node not in low_fp_ops:
                    if self.config["default_w8a16"]:
                        module.w_cfg.qspec.man_bit = 8
                        module.i_cfg.qspec.man_bit = 16
                    elif self.config["default_w8a8"]:
                        module.w_cfg.qspec.man_bit = 8
                        module.i_cfg.qspec.man_bit = 8
                    else:
                        if max_a_bit == 5:
                            module.i_cfg.qspec.man_bit = 8
                            module.i_cfg.qspec.max_exp_boost = 3
                        else:
                            module.i_cfg.qspec.man_bit = max_a_bit
                        module.w_cfg.qspec.man_bit = max_w_bit
                else:
                    # act bitwidth gets priority first and then weight bitwidth.
                    low_fp_ops[node].sort(key=lambda e: (e[2], e[1]))
                    module.w_cfg.qspec.man_bit = low_fp_ops[node][0][1]
                    if low_fp_ops[node][0][2] == 5:
                        module.i_cfg.qspec.max_exp_boost = 3
                        module.i_cfg.qspec.man_bit = 8
                    else:
                        module.i_cfg.qspec.man_bit = low_fp_ops[node][0][2]
                if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                    module.clear_quant_state()
                    module.fire_quant()
                    module.weight_static_quant()
                    module.fixed_quant()

                key = "w%da%d" % (
                    module.w_cfg.qspec.man_bit - module.w_cfg.qspec.max_exp_boost,
                    module.i_cfg.qspec.man_bit - module.i_cfg.qspec.max_exp_boost,
                )
                assert node.target not in statics[key]
                statics[key].add(node.target)
                visit.add(node.target)
            # for the rest, just set to w4a8.
            for i in range(num_high_fp_ops, len(sense_list)):
                node = sense_list[i][1]
                if node.target in visit:
                    continue
                module = self.module.get_submodule(node.target)
                module.w_cfg.qspec.man_bit = min(self.config["weight_bits"])  # 4)
                module.i_cfg.qspec.man_bit = min(self.config["act_bits"])  # 8
                if module.i_cfg.qspec.man_bit == 5:
                    module.i_cfg.qspec.max_exp_boost = 3
                    module.i_cfg.qspec.man_bit = 8
                key = "w%da%d" % (
                    module.w_cfg.qspec.man_bit - module.w_cfg.qspec.max_exp_boost,
                    module.i_cfg.qspec.man_bit - module.i_cfg.qspec.max_exp_boost,
                )
                assert node.target not in statics[key]
                statics[key].add(node.target)
                assert node.op == "call_module"
                if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
                    module.clear_quant_state()
                    module.fire_quant()
                    module.weight_static_quant()
                    module.fixed_quant()
                visit.add(node.target)
        elif self.config["policy"] == "bitwidth":
            raise NotImplementedError
        return statics
