# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_round.compressors.adam import AdamCompressor
    from auto_round.compressors.base import BaseCompressor, LLMCompressor
    from auto_round.compressors.config import (
        DiffusionExtraConfig,
        ExtraConfig,
        MLLMExtraConfig,
        SchemeExtraConfig,
        TuningExtraConfig,
    )
    from auto_round.compressors.diffusion.compressor import DiffusionCompressor
    from auto_round.compressors.diffusion.hybrid import HybridCompressor
    from auto_round.compressors.mllm.compressor import MLLMCompressor

__all__ = [
    "AdamCompressor",
    "BaseCompressor",
    "LLMCompressor",
    "MLLMCompressor",
    "DiffusionCompressor",
    "HybridCompressor",
    "DiffusionExtraConfig",
    "ExtraConfig",
    "MLLMExtraConfig",
    "SchemeExtraConfig",
    "TuningExtraConfig",
]


def __getattr__(name):
    if name == "AdamCompressor":
        from auto_round.compressors.adam import AdamCompressor

        return AdamCompressor

    if name in {"BaseCompressor", "LLMCompressor"}:
        from auto_round.compressors.base import BaseCompressor, LLMCompressor

        return {"BaseCompressor": BaseCompressor, "LLMCompressor": LLMCompressor}[name]

    if name == "MLLMCompressor":
        from auto_round.compressors.mllm.compressor import MLLMCompressor

        return MLLMCompressor

    if name == "DiffusionCompressor":
        from auto_round.compressors.diffusion.compressor import DiffusionCompressor

        return DiffusionCompressor

    if name == "HybridCompressor":
        from auto_round.compressors.diffusion.hybrid import HybridCompressor

        return HybridCompressor

    if name in {
        "DiffusionExtraConfig",
        "ExtraConfig",
        "MLLMExtraConfig",
        "SchemeExtraConfig",
        "TuningExtraConfig",
    }:
        from auto_round.compressors.config import (
            DiffusionExtraConfig,
            ExtraConfig,
            MLLMExtraConfig,
            SchemeExtraConfig,
            TuningExtraConfig,
        )

        return {
            "DiffusionExtraConfig": DiffusionExtraConfig,
            "ExtraConfig": ExtraConfig,
            "MLLMExtraConfig": MLLMExtraConfig,
            "SchemeExtraConfig": SchemeExtraConfig,
            "TuningExtraConfig": TuningExtraConfig,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
