import pytest

from gptqmodel.quantization import METHOD, QUANT_METHOD
from gptqmodel.quantization.config import AutoRoundVersion, FORMAT, QuantizeConfig


def test_quant_method_alias_exports_auto_round():
    assert QUANT_METHOD is METHOD
    assert METHOD.AUTO_ROUND == "auto_round"


def test_auto_round_quantize_config_defaults_method():
    cfg = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=128, sym=True, format=FORMAT.GPTQ)

    assert cfg.quant_method == METHOD.AUTO_ROUND
    assert cfg.group_size == 128
    assert cfg.format == FORMAT.GPTQ


def test_auto_round_quantize_config_derives_default_learning_rates():
    cfg = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=128, sym=True, format=FORMAT.GPTQ, iters=200)
    rtn_cfg = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=128, sym=True, format=FORMAT.GPTQ, iters=0)

    assert cfg.lr == pytest.approx(1.0 / 200)
    assert cfg.minmax_lr == pytest.approx(1.0 / 200)
    assert rtn_cfg.lr == pytest.approx(5e-3)
    assert rtn_cfg.minmax_lr == pytest.approx(5e-3)


def test_auto_round_quantize_config_standardizes_version_switch():
    cfg_v1 = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=128, sym=True, format=FORMAT.GPTQ)
    cfg_v2 = QuantizeConfig(auto_round_enabled=True, 
        bits=4,
        group_size=128,
        sym=True,
        format=FORMAT.GPTQ,
        auto_round_version=AutoRoundVersion.V2,
    )
    legacy_v2 = QuantizeConfig(auto_round_enabled=True, 
        bits=4,
        group_size=128,
        sym=True,
        format=FORMAT.GPTQ,
        enable_alg_ext=True,
    )

    assert cfg_v1.auto_round_version == AutoRoundVersion.V1
    assert cfg_v1.enable_alg_ext is False
    assert cfg_v2.auto_round_version == AutoRoundVersion.V2
    assert cfg_v2.enable_alg_ext is True
    assert cfg_v2.disable_opt_rtn is True
    assert legacy_v2.auto_round_version == AutoRoundVersion.V2


def test_auto_round_v2_requires_symmetric_quantization():
    with pytest.raises(ValueError, match="requires `sym=True`"):
        QuantizeConfig(auto_round_enabled=True, 
            bits=4,
            group_size=128,
            sym=False,
            format=FORMAT.GPTQ,
            auto_round_version=AutoRoundVersion.V2,
        )


def test_auto_round_quantize_config_serializes_gptq_storage_and_roundtrips():
    cfg = QuantizeConfig(auto_round_enabled=True, 
        bits=4,
        group_size=128,
        sym=False,
        format=FORMAT.GPTQ,
        hessian_mse=True,
        iters=123,
        lr=0.02,
        minmax_lr=0.01,
        enable_minmax_tuning=False,
        enable_alg_ext=True,
        not_use_best_mse=True,
        disable_opt_rtn=True,
        auto_round_version=AutoRoundVersion.V1,
    )

    payload = cfg.to_dict()
    loaded = QuantizeConfig.from_quant_config(payload)

    assert payload["quant_method"] == METHOD.GPTQ
    assert payload["checkpoint_format"] == FORMAT.GPTQ
    assert payload["sym"] is False
    assert payload["group_size"] == 128
    assert payload["meta"]["auto_round"] == {
        "auto_round_enabled": True,
        "auto_round_version": "v1",
        "iters": 123,
        "lr": 0.02,
        "minmax_lr": 0.01,
        "enable_minmax_tuning": False,
        "enable_alg_ext": False,
        "not_use_best_mse": True,
        "disable_opt_rtn": True,
    }
    assert loaded.quant_method == METHOD.AUTO_ROUND
    assert loaded.hessian_mse is True
    assert loaded.iters == 123
    assert loaded.auto_round_version == AutoRoundVersion.V1
    assert loaded.enable_alg_ext is False
    assert loaded.disable_opt_rtn is True


@pytest.mark.parametrize("group_size", [32, 64, 128, -1])
def test_quantize_config_accepts_common_group_sizes(group_size):
    cfg = QuantizeConfig(group_size=group_size)
    assert cfg.group_size == group_size
