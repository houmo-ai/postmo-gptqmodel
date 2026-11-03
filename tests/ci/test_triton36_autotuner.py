"""CPU contract for the Triton 3.6 autotuner cache patch."""

import importlib
import threading
from types import SimpleNamespace

from gptqmodel.utils import nogil_patcher


def test_triton36_autotuner_patch_initializes_missing_cache(monkeypatch):
    initialized = []

    class FakeAutotuner:
        _gptqmodel_threadsafe = False

        def __init__(self, *args, **kwargs):
            initialized.append((args, kwargs))
            self.cache = {}

    fake_module = SimpleNamespace(
        Autotuner=FakeAutotuner,
        builtins=__import__("builtins"),
        Config=object,
        driver=SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: None)),
        knobs=SimpleNamespace(autotuning=SimpleNamespace(print=False)),
        get_cache_manager=lambda key: None,
        triton_key=lambda: "key",
        get_cache_invalidating_env_vars=lambda: {},
        JITFunction=type("JITFunction", (), {}),
        hashlib=__import__("hashlib"),
        json=__import__("json"),
    )

    def fake_version(name):
        if name == "triton":
            return "3.6.0"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(nogil_patcher, "version", fake_version)
    monkeypatch.setattr(nogil_patcher.TritonPatch, "_applied", False)
    monkeypatch.setitem(__import__("sys").modules, "triton", SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "triton.runtime", SimpleNamespace(autotuner=fake_module))
    monkeypatch.setitem(__import__("sys").modules, "triton.runtime.autotuner", fake_module)

    nogil_patcher.patch_triton_autotuner()
    tuner = FakeAutotuner(1, flag=True)

    assert initialized
    assert tuner.cache == {}
    assert tuner._cache is tuner.cache
    assert isinstance(tuner._cache_lock, type(threading.RLock()))
    assert tuner._cache_futures == {}
    assert FakeAutotuner._gptqmodel_threadsafe is True
