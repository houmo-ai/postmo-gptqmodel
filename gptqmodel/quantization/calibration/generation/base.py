from typing import Dict


class CalibrationProviderBase:
    name = ""

    def generate(self, model, tokenizer, cfg) -> str:
        raise NotImplementedError()


_PROVIDERS: Dict[str, CalibrationProviderBase] = {}


def register_calibration_provider(provider: CalibrationProviderBase) -> None:
    key = str(provider.name).strip().lower()
    if not key:
        raise ValueError("Calibration generation: provider.name is required.")
    _PROVIDERS[key] = provider


def get_calibration_provider(name: str) -> CalibrationProviderBase:
    key = str(name).strip().lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Calibration generation: provider `{name}` is not registered.")
    return _PROVIDERS[key]