import copy
import importlib.util
import logging
import pathlib
import sys
import types
import unittest

import torch


def _load_hadamard_utils_module():
    # Avoid importing gptqmodel/__init__.py, which pulls heavy model deps during test collection.
    fake_gptqmodel = types.ModuleType("gptqmodel")
    fake_utils = types.ModuleType("gptqmodel.utils")
    fake_logger = types.ModuleType("gptqmodel.utils.logger")

    def setup_logger():
        return logging.getLogger("test_hadamard_fallback_compare")

    fake_logger.setup_logger = setup_logger
    fake_utils.logger = fake_logger
    fake_gptqmodel.utils = fake_utils

    sys.modules.setdefault("gptqmodel", fake_gptqmodel)
    sys.modules.setdefault("gptqmodel.utils", fake_utils)
    sys.modules.setdefault("gptqmodel.utils.logger", fake_logger)

    module_path = pathlib.Path(__file__).resolve().parents[1] / "gptqmodel/quantization/rotation/hadamard_utils.py"
    spec = importlib.util.spec_from_file_location("hadamard_utils_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


had_utils = _load_hadamard_utils_module()


class TestHadamardFallbackCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for this comparison test.")

        cls.fast_module = had_utils.import_fast_hadamard_transform(required=False)
        if cls.fast_module is None:
            raise unittest.SkipTest(
                "fast_hadamard_transform is unavailable in current environment; "
                "cannot compare fast path with fallback path."
            )

        cls.device = torch.device("cuda:0")
        torch.manual_seed(42)

    def _run_with_import_override(self, override_module, fn, *args, **kwargs):
        original = had_utils.import_fast_hadamard_transform
        had_utils.import_fast_hadamard_transform = lambda required=False: override_module
        try:
            return fn(*args, **kwargs)
        finally:
            had_utils.import_fast_hadamard_transform = original

    def _error_stats(self, ref: torch.Tensor, other: torch.Tensor):
        ref_f = ref.float()
        other_f = other.float()
        diff = (ref_f - other_f).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        max_rel = (diff / ref_f.abs().clamp_min(1e-12)).max().item()
        return max_abs, mean_abs, max_rel

    def test_matmul_hadu_cuda_fast_vs_fallback(self):
        # Cover K=1 and K>1 cases.
        for n in (1024, 344):
            with self.subTest(n=n):
                x = torch.randn(8, n, device=self.device, dtype=torch.float32)
                hadK, K = had_utils.get_hadK(n)

                y_fast = self._run_with_import_override(
                    self.fast_module, had_utils.matmul_hadU_cuda, x, hadK, K
                )
                y_fallback = self._run_with_import_override(
                    None, had_utils.matmul_hadU_cuda, x, hadK, K
                )

                max_abs, mean_abs, max_rel = self._error_stats(y_fast, y_fallback)
                print(
                    f"[hadamard-error][matmul_hadU_cuda][n={n},K={K}] "
                    f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}"
                )

                torch.testing.assert_close(y_fast, y_fallback, rtol=1e-4, atol=1e-4)

    def test_apply_exact_had_to_linear_chunked_fast_vs_fallback(self):
        # had_dim != -1 path in apply_exact_had_to_linear.
        for output in (True, False):
            with self.subTest(output=output):
                base = torch.nn.Linear(128, 256, bias=True, device=self.device, dtype=torch.float32)
                mod_fast = copy.deepcopy(base)
                mod_fallback = copy.deepcopy(base)

                self._run_with_import_override(
                    self.fast_module,
                    had_utils.apply_exact_had_to_linear,
                    mod_fast,
                    had_dim=64,
                    output=output,
                )
                self._run_with_import_override(
                    None,
                    had_utils.apply_exact_had_to_linear,
                    mod_fallback,
                    had_dim=64,
                    output=output,
                )

                w_max_abs, w_mean_abs, w_max_rel = self._error_stats(
                    mod_fast.weight.data, mod_fallback.weight.data
                )
                b_max_abs, b_mean_abs, b_max_rel = self._error_stats(
                    mod_fast.bias.data, mod_fallback.bias.data
                )
                print(
                    f"[hadamard-error][apply_exact_had_to_linear][had_dim=64,output={output}] "
                    f"weight(max_abs={w_max_abs:.6e}, mean_abs={w_mean_abs:.6e}, max_rel={w_max_rel:.6e}), "
                    f"bias(max_abs={b_max_abs:.6e}, mean_abs={b_mean_abs:.6e}, max_rel={b_max_rel:.6e})"
                )

                torch.testing.assert_close(
                    mod_fast.weight.data, mod_fallback.weight.data, rtol=1e-4, atol=1e-4
                )
                torch.testing.assert_close(
                    mod_fast.bias.data, mod_fallback.bias.data, rtol=1e-4, atol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
