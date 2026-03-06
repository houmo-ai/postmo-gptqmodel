# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from gptqmodel.quantization import METHOD, QuantizeConfig


def test_act_group_aware_enabled_by_default_for_gptq():
    cfg = QuantizeConfig()
    assert cfg.quant_method == METHOD.GPTQ
    assert cfg.act_group_aware is True
    assert cfg.desc_act is False


def test_desc_act_enabling_auto_disables_act_group_aware(capfd):
    with pytest.raises(AssertionError):
        QuantizeConfig(desc_act=True)


def test_explicit_desc_act_and_act_group_aware_raises():
    with pytest.raises((ValueError, AssertionError)):
        QuantizeConfig(desc_act=True, act_group_aware=True)
