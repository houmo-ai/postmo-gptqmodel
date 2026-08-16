from __future__ import annotations

import pytest
import torch
from torch import nn

from gptqmodel.utils.structure import (
    alias_all_from_turtle_if_meta,
    alias_from_turtle_for_submodule,
)


class _NoMetaShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"tie_word_embeddings": False})()
        self.fc = nn.Linear(2, 2)


class _MetaShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"tie_word_embeddings": False})()
        self.fc = nn.Linear(2, 2, device="meta")


class _ExplodingTurtle(nn.Module):
    def named_modules(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("turtle named_modules should not be built without shell meta targets")


class _ExplodingTurtleWithLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def named_modules(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("turtle named_modules should not be built for targeted meta sync")


def test_alias_all_skips_turtle_map_when_shell_has_no_meta_leaf():
    shell = _NoMetaShell()

    assert alias_all_from_turtle_if_meta(shell, _ExplodingTurtle()) == 0


def test_alias_all_replaces_meta_leaf_from_turtle():
    shell = _MetaShell()
    turtle = _NoMetaShell()

    assert shell.fc.weight.device.type == "meta"
    assert alias_all_from_turtle_if_meta(shell, turtle) == 1
    assert shell.fc is turtle.fc
    assert shell.fc.weight.device.type == "cpu"


def test_alias_all_uses_targeted_turtle_lookup_for_meta_leaf():
    shell = _MetaShell()
    turtle = _ExplodingTurtleWithLeaf()

    assert alias_all_from_turtle_if_meta(shell, turtle) == 1
    assert shell.fc is turtle.fc


def test_alias_submodule_resolves_its_path_from_target_identity():
    shell = _MetaShell()
    turtle = _NoMetaShell()
    with torch.no_grad():
        turtle.fc.weight.fill_(3.0)
        turtle.fc.bias.fill_(4.0)

    result = alias_from_turtle_for_submodule(
        target_model=shell,
        turtle_model=turtle,
        target_submodule=shell.fc,
        device=torch.device("cpu"),
    )

    assert result is shell.fc
    assert result.weight.device.type == "cpu"
    assert torch.equal(result.weight, turtle.fc.weight)
    assert torch.equal(result.bias, turtle.fc.bias)


def test_alias_submodule_uses_targeted_turtle_lookup():
    shell = _MetaShell()
    turtle = _ExplodingTurtleWithLeaf()

    alias_from_turtle_for_submodule(
        target_model=shell,
        turtle_model=turtle,
        target_submodule=shell.fc,
        device=torch.device("cpu"),
    )

    assert shell.fc.weight.device.type == "cpu"


def test_alias_submodule_rejects_module_outside_target_model():
    with pytest.raises(ValueError, match="Could not find the given module"):
        alias_from_turtle_for_submodule(
            target_model=_MetaShell(),
            turtle_model=_NoMetaShell(),
            target_submodule=nn.Linear(2, 2, device="meta"),
            device=torch.device("cpu"),
        )


def test_alias_submodule_reports_target_path_missing_from_turtle():
    shell = _MetaShell()
    turtle = nn.Module()

    with pytest.raises(KeyError, match="fc.*target_model.*turtle_model"):
        alias_from_turtle_for_submodule(
            target_model=shell,
            turtle_model=turtle,
            target_submodule=shell.fc,
            device=torch.device("cpu"),
        )
