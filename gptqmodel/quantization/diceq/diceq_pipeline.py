from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from ...utils.logger import setup_logger
from .data_utils import get_loaders
from .get_rotation_matrix import train_rotation_for_targets
from .get_train_data_hook import collect_training_data_for_diceq


log = setup_logger()


def _resolve_diceq_pipeline_config(model) -> Optional[Dict[str, Any]]:
    resolver = getattr(model, "get_diceq_pipeline_config", None)
    if not callable(resolver):
        return None
    cfg = resolver()
    if not isinstance(cfg, dict):
        return None
    if not bool(cfg.get("enabled", False)):
        return None
    return cfg


def _check_rotation_cache(
    artifacts_root: Path,
    layer_config: Dict[str, Any],
    selected_targets: List[str],
) -> Optional[Dict[str, Any]]:
    """
    检查是否存在有效的旋转矩阵缓存。

    遍历每个 target 的 artifacts 目录，寻找最新的 .pt 文件，
    加载其中的 rotation_matrices 字段并提取可用的旋转矩阵。

    Returns:
        如果所有 target 都有有效缓存，返回与 train_rotation_for_targets 相同格式的字典；
        否则返回 None。
    """
    if not artifacts_root.exists():
        return None

    result: Dict[str, Any] = {}
    for target_name in selected_targets:
        target_dir = artifacts_root / target_name
        if not target_dir.exists():
            log.info(f"DiceQ: 旋转矩阵缓存目录不存在: {target_dir}")
            return None

        # 找到该 target 下最新的 .pt 文件（按文件名排序，时间戳在文件名末尾）
        pt_files = sorted(target_dir.glob("*.pt"))
        if not pt_files:
            log.info(f"DiceQ: target '{target_name}' 无缓存的旋转矩阵文件")
            return None

        latest_pt = pt_files[-1]  # 最新文件
        payload = torch.load(latest_pt, map_location="cpu", weights_only=False)
        rotation_dict = payload.get("rotation_matrices")
        if not isinstance(rotation_dict, dict) or not rotation_dict:
            log.warn(f"DiceQ: 缓存文件 '{latest_pt}' 中无有效的 rotation_matrices")
            return None

        # 使用与 train_rotation_for_target 相同的提取逻辑
        if target_name in rotation_dict and len(rotation_dict) == 1:
            extracted = rotation_dict[target_name]
        else:
            extracted = rotation_dict

        result[target_name] = extracted
        log.info(f"DiceQ: 从缓存加载 target '{target_name}' 的旋转矩阵: {latest_pt.name}")

    return result


def _check_activation_cache(
    data_root: Path,
    layer_config: Dict[str, Any],
    expected_nsamples: int,
) -> bool:
    """
    检查是否存在有效的激活数据缓存。

    检查逻辑：
    1. 检查 data_root 目录是否存在
    2. 检查每个 r_type 对应的文件夹是否存在且非空
    3. 检查每个文件夹中的样本数量是否满足要求

    Args:
        data_root: 激活数据根目录
        layer_config: 层配置字典
        expected_nsamples: 期望的样本数量

    Returns:
        bool: 缓存是否有效
    """
    if not data_root.exists():
        log.debug(f"激活数据缓存目录不存在: {data_root}")
        return False

    for r_type, cfg in layer_config.items():
        if not isinstance(cfg, dict):
            continue

        folder = cfg.get("folder", r_type)
        r_type_path = data_root / folder

        if not r_type_path.exists():
            log.debug(f"r_type '{r_type}' 的缓存目录不存在: {r_type_path}")
            return False

        # 检查目录是否为空
        files = list(r_type_path.iterdir())
        if not files:
            log.debug(f"r_type '{r_type}' 的缓存目录为空: {r_type_path}")
            return False

        # 检查样本数量（根据保存模式判断）
        save_mode = cfg.get("save_mode", "separate")
        if save_mode == "separate":
            # 逐个样本保存模式：检查文件数量
            # 每个样本可能有多个文件（多个层），所以文件数应该 >= expected_nsamples
            if len(files) < expected_nsamples:
                log.debug(f"r_type '{r_type}' 的缓存样本数量不足: {len(files)} < {expected_nsamples}")
                return False
        elif save_mode == "combined":
            # 合并保存模式：检查是否存在合并后的文件
            # 合并模式通常只有一个文件，包含所有样本
            if len(files) < 1:
                log.debug(f"r_type '{r_type}' 的合并缓存文件不存在")
                return False

    log.info(
        f"激活数据缓存检查通过，所有 "
        f"{len([k for k in layer_config.keys() if isinstance(layer_config[k], dict)])} 个 r_type 的缓存均有效"
    )
    return True


def run_diceq_pipeline(
    *,
    model,
    rotation_device: Union[str, torch.device],
) -> Optional[Dict[str, Any]]:
    diceq_cfg = _resolve_diceq_pipeline_config(model)
    if diceq_cfg is None:
        return None

    log.info("DiceQ: enabled, preparing data and training rotation matrices.")

    layer_config = diceq_cfg.get("layer_config")
    if not isinstance(layer_config, dict) or not layer_config:
        raise ValueError("DiceQ config must provide non-empty `layer_config` dict.")

    seqlen = int(diceq_cfg.get("seqlen", 2048))
    collect_nsamples = int(diceq_cfg.get("nsamples", 128))
    seed = int(diceq_cfg.get("seed", 0))
    calib_dataset = diceq_cfg.get("calib_dataset", "wikitext2")
    targets = diceq_cfg.get("targets")
    if isinstance(targets, str):
        targets = [targets]
    elif targets is not None:
        targets = list(targets)

    # device 用于激活数据采集，从 rotation_device 自动推导
    if isinstance(rotation_device, torch.device):
        device_arg = rotation_device.type
    else:
        device_arg = str(rotation_device).lower()
        if device_arg in {"device.cpu", "cpu"}:
            device_arg = "cpu"
        elif "cuda" in device_arg:
            device_arg = "cuda"
        else:
            device_arg = "auto"

    # 获取临时数据目录配置，默认为 ./diceq_tmp
    tmp_data_dir = diceq_cfg.get("tmp_data_dir", "./diceq_tmp")
    model_type = getattr(model.model.config, "model_type", "unknown_model")
    tmp_root = Path(tmp_data_dir) / model_type
    tmp_root.mkdir(parents=True, exist_ok=True)
    data_root = tmp_root / "data"
    train_root = tmp_root / "rotation"
    log.info(f"DiceQ: 使用模型专属临时数据目录: {tmp_root.absolute()}")

    # Determine data types and debug mode from config
    needs_activation = False
    needs_weight = False
    debug_mode = diceq_cfg.get("debug", False)

    for r_type, cfg in layer_config.items():
        if isinstance(cfg, dict):
            data_type = cfg.get("data_type", "activation")
            if data_type == "activation":
                needs_activation = True
            elif data_type == "weight":
                needs_weight = True
            # Check debug flag in config entry (entry-level overrides global)
            if cfg.get("debug") is not None:
                debug_mode = cfg.get("debug")

    # 检查是否已存在数据缓存（激活或权重）
    cache_valid = False
    if needs_activation:
        cache_valid = _check_activation_cache(data_root, layer_config, collect_nsamples)
        if cache_valid:
            log.info("DiceQ: 发现有效的激活数据缓存，跳过激活搜集步骤")
    # Note: Weight data doesn't need cache check as it's fast to collect

    if cache_valid and needs_activation:
        # 直接使用缓存的数据路径
        target_data_path = {}
        for r_type, cfg in layer_config.items():
            if isinstance(cfg, dict):
                folder = cfg.get("folder", r_type)
                target_data_path[r_type] = str(data_root / folder)
    else:
        if needs_activation:
            log.info("DiceQ: 未找到有效的激活数据缓存，开始搜集激活数据")
            # 使用 get_loaders 内部获取校准数据集
            diceq_calibration = get_loaders(
                calib_dataset,
                nsamples=collect_nsamples,
                seed=seed,
                model=model.model.name_or_path if hasattr(model.model, "name_or_path") else model.model.config._name_or_path,
                seqlen=seqlen,
                eval_mode=False,
            )
        else:
            diceq_calibration = None
            log.info("DiceQ: 权重模式，无需校准数据集")

        target_data_path = collect_training_data_for_diceq(
            model=model.model,
            dataloader=diceq_calibration,
            output_root=data_root,
            r_config=layer_config,
            nsamples=collect_nsamples,
            seqlen=seqlen,
            seed=seed,
            device=device_arg,
        )
        if needs_activation:
            log.info("DiceQ: 激活数据搜集完成")
        if needs_weight:
            log.info("DiceQ: 权重数据搜集完成")

    # 检查旋转矩阵缓存
    selected_targets = targets if targets else [k for k, v in layer_config.items() if isinstance(v, dict)]
    rotation_cache = _check_rotation_cache(
        train_root / "artifacts", layer_config, selected_targets
    )
    if rotation_cache is not None:
        log.info("DiceQ: 发现有效的旋转矩阵缓存，跳过训练步骤")
        return rotation_cache

    target_rotations = train_rotation_for_targets(
        target_to_data_path=target_data_path,
        config_dict=layer_config,
        targets=targets,
        seed=seed,
        model=diceq_cfg.get("model"),
        hf_token=diceq_cfg.get("hf_token"),
        head_num=diceq_cfg.get("head_num"),
        log_dir=train_root / "logs",
        log_level=str(diceq_cfg.get("log_level", "INFO")),
        output_dir=train_root / "artifacts",
        debug=debug_mode,
    )
    log.info("DiceQ: rotation matrices ready.")
    return target_rotations


