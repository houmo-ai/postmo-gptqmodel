import pytest
import torch
from accelerate import init_empty_weights
from torch import nn

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import create_quant_module


class _LinearShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)


def _replace_projection(shell: nn.Module, *, allow_meta: bool) -> None:
    create_quant_module(
        name="proj",
        linear_cls=TorchQuantLinear,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=64,
        module=shell,
        submodule=shell.proj,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.TORCH,
        allow_meta=allow_meta,
    )


def test_quant_module_creation_rejects_meta_by_default():
    with init_empty_weights(include_buffers=True):
        shell = _LinearShell()

    with pytest.raises(AssertionError, match="Expected `proj` to reside on CPU"):
        _replace_projection(shell, allow_meta=False)


def test_quant_module_creation_keeps_packed_shell_on_meta_when_enabled():
    with init_empty_weights(include_buffers=True):
        shell = _LinearShell()
        _replace_projection(shell, allow_meta=True)

    assert isinstance(shell.proj, TorchQuantLinear)
    assert shell.proj.qweight.device.type == "meta"
    assert shell.proj.qzeros.device.type == "meta"
    assert shell.proj.scales.device.type == "meta"
    assert shell.proj.g_idx.device.type == "meta"
