import pytest
import torch

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase, _global_tracker, safe_to_cpu_
from auto_round.utils import global_state
from auto_round.utils.model import mv_module_from_gpu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class ParentReportsCpu(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.child = torch.nn.Linear(2, 2, device="cuda")
        self.register_buffer("cuda_buffer", torch.ones(1, device="cuda"))
        self.register_parameter("meta_weight", torch.nn.Parameter(torch.empty(1, device="meta")))


class OriginalTrackedModule(ParentReportsCpu):
    pass


class ReplacementWithoutOriginalChild(ReplacementModuleBase):
    @classmethod
    def original_module_class(cls):
        return "OriginalTrackedModule"

    @classmethod
    def from_original(cls, original, config=None):
        return cls(original)


def assert_offloaded_to_cpu_with_meta_preserved(module):
    assert module.child.weight.device.type == "cpu"
    assert module.child.bias.device.type == "cpu"
    assert module.cuda_buffer.device.type == "cpu"
    assert module.meta_weight.device.type == "meta"


def test_mv_module_from_gpu_recurses_when_parent_reports_cpu():
    module = ParentReportsCpu()

    mv_module_from_gpu(module)

    assert_offloaded_to_cpu_with_meta_preserved(module)


def test_safe_to_cpu_offloads_when_replacements_exist():
    module = ParentReportsCpu()
    previous_count = global_state.replaced_module_count
    global_state.replaced_module_count = 1
    try:
        safe_to_cpu_(module)
    finally:
        global_state.replaced_module_count = previous_count

    assert_offloaded_to_cpu_with_meta_preserved(module)


def test_safe_to_cpu_offloads_tracker_originals_when_replacements_exist():
    original = OriginalTrackedModule()
    replacement = ReplacementWithoutOriginalChild.from_original(original)
    container = torch.nn.Module()
    container.replacement = replacement
    previous_count = global_state.replaced_module_count
    global_state.replaced_module_count = 1
    try:
        safe_to_cpu_(container)
        tracked_original = _global_tracker.get_original(replacement)
        assert tracked_original is original
        assert_offloaded_to_cpu_with_meta_preserved(tracked_original)
    finally:
        _global_tracker.clear()
        global_state.replaced_module_count = previous_count
