from __future__ import annotations

from torch import nn

from gptqmodel.utils.structure import alias_all_from_turtle_if_meta


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
