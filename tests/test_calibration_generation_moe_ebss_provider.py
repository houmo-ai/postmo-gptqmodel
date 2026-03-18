# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.quantization.calibration.generation.moe_ebss_provider import MoeEbssProvider


def test_moe_ebss_provider_name():
    provider = MoeEbssProvider()
    assert provider.name == "moe_ebss"
