"""
train_oot.py
===============================================================================

OOT（Occluded Object Transport）实验的第一版训练入口。

该文件借鉴原仓库 train.py 的整体工程结构，但做了四类 OOT 专用修改：

1. 模型前向过程拆为：
       encode -> remember -> predict

2. 在普通 next-embedding MSE 之外，额外统计：
       - 全时段 MSE
       - 遮挡时段 MSE
       - 可见时段 MSE
       - 再现时刻 MSE
       - cosine prediction error
       - memory hidden norm / memory change
       - Leaky-RNN leak rate

3. metadata（phase_id、episode_id 等）不做连续值归一化。

4. 可选地把少量 validation latent 保存为 .npz，
   为后续 OOT 工作记忆与 schema 分析保留中间结果。

重要假设
--------
A. 数据仍通过 stable_worldmodel.data.load_dataset 加载；
B. batch 至少包含：
       pixels
       action
C. 推荐额外包含：
       phase_id
       object_visible / is_visible / visible
       is_reappearance
       episode_id
       timestep
D. 每个训练样本的时间长度通常为：
       history_size + num_preds

当前代码默认 num_preds=1，与原 LeWM 配置一致。
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

# 继续使用原论文代码中的 SIGReg，保证正则项与基线可比。
from module import SIGReg

# 继续使用原项目提供的图像预处理、连续列归一化和 checkpoint callback。
from utils import SaveCkptCallback, get_column_normalizer, get_img_preprocessor


# 这些字段是“离散标签或索引”，不应该像 action/proprio 一样做均值方差归一化。
DEFAULT_METADATA_KEYS = {
    "episode_id",
    "timestep",
    "phase_id",
    "schema_id",
    "content_id",
    "cue_id",
    "target_id",
    "shown_target_id",
    "is_violation",
    "object_visible",
    "is_visible",
    "visible",
    "is_reappearance",
    "valid_mask",
    "episode_start",
    "reset_mask",
}


def cfg_select(cfg: DictConfig, path: str, default: Any) -> Any:
    """
    安全读取可选 Hydra 配置。

    例如：
        cfg_select(cfg, "oot_loss.reappearance_weight", 3.0)

    即使 oot_loss 字段尚未写入 YAML，第一版代码也能使用默认值运行。
    """
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def first_existing(
    mapping: Dict[str, Tensor],
    candidate_names: Sequence[str],
) -> Optional[Tensor]:
    """从若干候选 key 中返回 batch 实际拥有的第一个。"""
    for name in candidate_names:
        if name in mapping:
            return mapping[name]
    return None


def squeeze_last_singleton(x: Tensor) -> Tensor:
    """把 (B,T,1) 标签简化为 (B,T)，其他形状保持不变。"""
    if x.ndim >= 1 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


def align_target_time(
    x: Optional[Tensor],
    num_preds: int,
    context_length: int,
) -> Optional[Tensor]:
    """
    把逐帧 metadata 与 latent target 对齐。

    当 num_preds=1 时：
        predictor 输入时刻：0, 1, ..., T-2
        target 时刻：       1, 2, ..., T-1

    因而 metadata 也要从第 1 帧开始切片。
    """
    if x is None:
        return None

    x = squeeze_last_singleton(x)

    if x.ndim < 2:
        raise ValueError(
            "逐时刻 metadata 至少应为 (B,T)，"
            f"实际 shape={tuple(x.shape)}。"
        )

    end = num_preds + context_length

    if x.size(1) < end:
        raise ValueError(
            f"metadata 时间长度={x.size(1)}，不足以切取 [{num_preds}:{end}]。"
        )

    return x[:, num_preds:end]


def as_float_mask(
    x: Optional[Tensor],
    reference: Tensor,
) -> Optional[Tensor]:
    """把 bool/int/float 标签转换为与 reference 同 device/dtype 的 0-1 mask。"""
    if x is None:
        return None
    return x.to(device=reference.device, dtype=reference.dtype)


def values_to_mask(
    labels: Optional[Tensor],
    values: Iterable[int],
    reference: Tensor,
) -> Optional[Tensor]:
    """构造 labels ∈ values 的 mask。"""
    if labels is None:
        return None

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for value in values:
        mask |= labels == int(value)

    return mask.to(device=reference.device, dtype=reference.dtype)


def safe_masked_mean(
    values: Tensor,
    mask: Optional[Tensor],
) -> Tensor:
    """
    对 mask 指定的元素求均值。

    若当前 batch 中 mask 没有任何有效元素，返回 0，而不是 NaN。
    """
    if mask is None:
        return values.mean()

    mask = mask.to(device=values.device, dtype=values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum().clamp_min(1.0)
    return numerator / denominator


def get_sequence_masks(
    batch: Dict[str, Tensor],
    num_preds: int,
    context_length: int,
    step_mse: Tensor,
    cfg: DictConfig,
) -> Dict[str, Optional[Tensor]]:
    """
    从 batch metadata 中构造 OOT 分阶段 mask。

    返回的每个 mask 都与 step_mse 同形状 (B,T_context)。
    """
    # ---------- 有效时间步 ----------
    valid_raw = first_existing(batch, ["valid_mask"])
    valid_target = align_target_time(valid_raw, num_preds, context_length)
    valid_mask = as_float_mask(valid_target, step_mse)

    if valid_mask is None:
        valid_mask = torch.ones_like(step_mse)

    # ---------- 可见 / 遮挡 ----------
    visible_raw = first_existing(
        batch,
        ["object_visible", "is_visible", "visible"],
    )
    visible_target = align_target_time(
        visible_raw,
        num_preds,
        context_length,
    )
    visible_mask = as_float_mask(visible_target, step_mse)

    occluded_mask = None
    if visible_mask is not None:
        visible_mask = visible_mask * valid_mask
        occluded_mask = (1.0 - visible_mask) * valid_mask

    # ---------- phase_id ----------
    phase_raw = first_existing(batch, ["phase_id"])
    phase_target = align_target_time(
        phase_raw,
        num_preds,
        context_length,
    )

    # ---------- 再现时刻 ----------
    # 优先使用数据中直接给出的 is_reappearance；
    # 若没有，则根据 phase_id 和配置中的 phase 编号推断。
    reappearance_raw = first_existing(batch, ["is_reappearance"])
    reappearance_target = align_target_time(
        reappearance_raw,
        num_preds,
        context_length,
    )
    reappearance_mask = as_float_mask(reappearance_target, step_mse)

    if reappearance_mask is None and phase_target is not None:
        phase_ids = cfg_select(
            cfg,
            "oot_loss.reappearance_phase_ids",
            [3],
        )
        reappearance_mask = values_to_mask(
            phase_target,
            phase_ids,
            step_mse,
        )

    if reappearance_mask is not None:
        reappearance_mask = reappearance_mask * valid_mask

    return {
        "valid": valid_mask,
        "visible": visible_mask,
        "occluded": occluded_mask,
        "reappearance": reappearance_mask,
        "phase_id": phase_target,
    }


def oot_forward(
    self,
    batch: Dict[str, Tensor],
    stage: str,
    cfg: DictConfig,
) -> Dict[str, Tensor]:
    """
    OOT 专用前向、损失与日志函数。

    self 是 stable_pretraining.Module；
    self.model 是 Hydra 实例化得到的 module_oot.OOTJEPA。
    """
    context_length = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    sigreg_weight = float(cfg.loss.sigreg.weight)

    if num_preds != 1:
        # 当前 predictor 的目标对齐主要按单步预测设计。
        # 保留显式报错，避免 num_preds 改大后静默产生错误标签。
        raise NotImplementedError(
            "train_oot.py 初版仅支持 num_preds=1。"
        )

    if "action" not in batch:
        raise KeyError("OOT 训练 batch 必须包含 'action'。")

    # HDF5 序列边界附近可能出现 NaN action。
    # 与原 train.py 一致，用 0 替换，避免 NaN 传播到整个网络。
    clean_batch = dict(batch)
    clean_batch["action"] = torch.nan_to_num(
        batch["action"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # 1. encode：每帧像素 -> 视觉 latent；动作 -> action latent
    # ------------------------------------------------------------------
    encoded = self.model.encode(clean_batch)

    emb = encoded["emb"]          # (B, T_total, D)
    act_emb = encoded["act_emb"]  # (B, T_total, D_action)

    expected_total_length = context_length + num_preds
    if emb.size(1) < expected_total_length:
        raise ValueError(
            f"模型收到的序列长度={emb.size(1)}，"
            f"但 history_size + num_preds={expected_total_length}。"
        )

    # predictor 输入 0 ... T-2，target 是 1 ... T-1。
    context_emb = emb[:, :context_length]
    context_action = act_emb[:, :context_length]
    target_emb = emb[:, num_preds:num_preds + context_length]

    # ------------------------------------------------------------------
    # 2. remember：视觉 latent -> Leaky-RNN 工作记忆
    # ------------------------------------------------------------------
    # valid_mask/reset_mask 是针对“输入时刻”的，而不是 target 时刻。
    input_valid_mask = first_existing(clean_batch, ["valid_mask"])
    if input_valid_mask is not None:
        input_valid_mask = squeeze_last_singleton(input_valid_mask)
        input_valid_mask = input_valid_mask[:, :context_length]

    input_reset_mask = first_existing(
        clean_batch,
        ["reset_mask", "episode_start"],
    )
    if input_reset_mask is not None:
        input_reset_mask = squeeze_last_singleton(input_reset_mask)
        input_reset_mask = input_reset_mask[:, :context_length]

    memory_emb, memory_hidden, final_memory = self.model.remember(
        context_emb,
        context_action,
        valid_mask=input_valid_mask,
        reset_mask=input_reset_mask,
    )

    # ------------------------------------------------------------------
    # 3. predict：基于工作记忆预测下一时刻 latent
    # ------------------------------------------------------------------
    pred_emb = self.model.predict(memory_emb, context_action)

    if pred_emb.shape != target_emb.shape:
        raise RuntimeError(
            "预测与 target 的形状不一致："
            f"pred={tuple(pred_emb.shape)}, target={tuple(target_emb.shape)}。"
        )

    # 每个 batch、每个预测时刻独立计算 latent MSE。
    # 对 embedding dimension 求均值后得到 (B,T_context)。
    step_mse = (pred_emb - target_emb).pow(2).mean(dim=-1)

    masks = get_sequence_masks(
        clean_batch,
        num_preds,
        context_length,
        step_mse,
        cfg,
    )

    all_pred_loss = safe_masked_mean(step_mse, masks["valid"])

    # OOT 的关键检验通常是对象重新出现时预测是否正确。
    # 初版允许提高 reappearance 时刻权重，但仍保留所有时刻的监督。
    reappearance_weight = float(
        cfg_select(cfg, "oot_loss.reappearance_weight", 3.0)
    )

    prediction_weights = masks["valid"].clone()

    if masks["reappearance"] is not None:
        prediction_weights = prediction_weights + (
            reappearance_weight - 1.0
        ) * masks["reappearance"]

    weighted_pred_loss = safe_masked_mean(
        step_mse * prediction_weights,
        masks["valid"],
    )

    # 原 LeWM 的第二个核心损失：让 encoder latent 接近各向同性高斯分布。
    sigreg_loss = self.sigreg(emb.transpose(0, 1))

    total_loss = weighted_pred_loss + sigreg_weight * sigreg_loss

    # ------------------------------------------------------------------
    # 4. 诊断指标
    # ------------------------------------------------------------------
    # cosine error 只作为指标，不代替原始 MSE。
    cosine_error_per_step = 1.0 - F.cosine_similarity(
        pred_emb,
        target_emb,
        dim=-1,
    )
    cosine_error = safe_masked_mean(
        cosine_error_per_step,
        masks["valid"],
    )

    # memory_hidden 的绝对尺度。
    memory_hidden_norm = memory_hidden.norm(dim=-1).mean()

    # 相邻时刻 memory embedding 的平均变化量。
    # 后续可分 phase 分析：遮挡期间是否保持/更新了合理状态。
    if memory_emb.size(1) > 1:
        memory_step_change = (
            memory_emb[:, 1:] - memory_emb[:, :-1]
        ).pow(2).mean()
    else:
        memory_step_change = memory_emb.new_zeros(())

    output: Dict[str, Tensor] = {
        "loss": total_loss,
        "pred_loss": weighted_pred_loss,
        "pred_loss_all": all_pred_loss,
        "sigreg_loss": sigreg_loss,
        "cosine_error": cosine_error,
        "memory_hidden_norm": memory_hidden_norm,
        "memory_step_change": memory_step_change,
        "leak_rate": self.model.leak_rate,
    }

    # 只有相应标签存在时才记录分阶段 MSE。
    if masks["visible"] is not None:
        output["pred_loss_visible"] = safe_masked_mean(
            step_mse,
            masks["visible"],
        )

    if masks["occluded"] is not None:
        output["pred_loss_occluded"] = safe_masked_mean(
            step_mse,
            masks["occluded"],
        )

    if masks["reappearance"] is not None:
        output["pred_loss_reappearance"] = safe_masked_mean(
            step_mse,
            masks["reappearance"],
        )

    # Lightning 日志只记录标量，避免把大张量发送到 logger。
    scalar_logs = {
        f"{stage}/{name}": value.detach()
        for name, value in output.items()
        if value.ndim == 0
    }

    self.log_dict(
        scalar_logs,
        on_step=True,
        on_epoch=True,
        prog_bar=name_in_progress_bar(stage, scalar_logs),
        sync_dist=True,
        batch_size=emb.size(0),
    )

    # ------------------------------------------------------------------
    # 5. 返回少量中间量，供 validation callback 保存
    # ------------------------------------------------------------------
    # 这些张量仍与计算图相连；callback 保存时会 detach。
    output.update(
        {
            "emb": emb,
            "context_emb": context_emb,
            "memory_emb": memory_emb,
            "memory_hidden": memory_hidden,
            "final_memory": final_memory,
            "pred_emb": pred_emb,
            "target_emb": target_emb,
            "step_mse": step_mse,
        }
    )

    if masks["phase_id"] is not None:
        output["target_phase_id"] = masks["phase_id"]

    for mask_name in ["valid", "visible", "occluded", "reappearance"]:
        mask = masks[mask_name]
        if mask is not None:
            output[f"target_{mask_name}_mask"] = mask

    return output


def name_in_progress_bar(
    stage: str,
    scalar_logs: Dict[str, Tensor],
) -> bool:
    """
    控制进度条是否展示这些指标。

    Lightning 的 log_dict 只能统一设置 prog_bar；
    为避免进度条过长，这里仅在 train/val 的标量数量较小时开启。
    """
    return len(scalar_logs) <= 4 and stage in {"train", "val"}


class OOTLatentDumpCallback(pl.Callback):
    """
    每隔若干 epoch 保存少量 validation latent 为 .npz。

    目的不是保存整个数据集，而是保留足以做初步可视化的样本：
        - encoder latent
        - Leaky-RNN hidden state
        - memory embedding
        - predictor output
        - target embedding
        - phase / visibility / reappearance 标签
    """

    def __init__(
        self,
        save_dir: str | Path,
        every_n_epochs: int = 1,
        max_batches_per_epoch: int = 4,
        max_samples_per_batch: int = 8,
    ) -> None:
        super().__init__()

        self.save_dir = Path(save_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_batches_per_epoch = max(1, int(max_batches_per_epoch))
        self.max_samples_per_batch = max(1, int(max_samples_per_batch))
        self.saved_this_epoch = 0

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.saved_this_epoch = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Tensor]],
        batch: Dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, dataloader_idx

        # 不保存 Lightning 在正式训练前运行的 sanity validation。
        if trainer.sanity_checking:
            return

        # 多 GPU 时只让 global rank 0 写文件，避免重复和文件冲突。
        if not trainer.is_global_zero:
            return

        if outputs is None:
            return

        epoch = int(trainer.current_epoch)

        if epoch % self.every_n_epochs != 0:
            return

        if self.saved_this_epoch >= self.max_batches_per_epoch:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        tensor_keys = [
            "emb",
            "context_emb",
            "memory_emb",
            "memory_hidden",
            "final_memory",
            "pred_emb",
            "target_emb",
            "step_mse",
            "target_phase_id",
            "target_valid_mask",
            "target_visible_mask",
            "target_occluded_mask",
            "target_reappearance_mask",
        ]

        payload: Dict[str, np.ndarray] = {}

        for key in tensor_keys:
            value = outputs.get(key)

            if not torch.is_tensor(value):
                continue

            detached = value.detach().cpu()

            # 对带 batch 维的张量只保存前几个样本。
            if detached.ndim >= 1 and detached.shape[0] == batch["pixels"].shape[0]:
                detached = detached[:self.max_samples_per_batch]

            payload[key] = detached.numpy()

        # 同时保存原 batch 中对分析有用、但不占太大空间的 metadata。
        metadata_keys = [
            "episode_id",
            "timestep",
            "phase_id",
            "schema_id",
            "content_id",
            "object_visible",
            "is_visible",
            "visible",
            "is_reappearance",
            "object_x",
            "object_y",
            "object_position",
            "action",
        ]

        for key in metadata_keys:
            value = batch.get(key)

            if not torch.is_tensor(value):
                continue

            detached = value.detach().cpu()

            if detached.ndim >= 1 and detached.shape[0] == batch["pixels"].shape[0]:
                detached = detached[:self.max_samples_per_batch]

            payload[f"batch_{key}"] = detached.numpy()

        filename = (
            f"epoch_{epoch:04d}_"
            f"val_batch_{batch_idx:04d}.npz"
        )

        np.savez_compressed(self.save_dir / filename, **payload)
        self.saved_this_epoch += 1


@hydra.main(
    version_base=None,
    config_path="./config/train",
    # 暂时继续使用原 lewm.yaml 作为总配置。
    # 后续只需新增 data=oot 与 model=oot_leaky 两个配置组。
    config_name="lewm",
)
def run(cfg: DictConfig) -> None:
    """构造数据、模型、优化器和 Lightning trainer。"""

    # ==================================================================
    # 1. Dataset
    # ==================================================================
    dataset_cfg = OmegaConf.to_container(
        cfg.data.dataset,
        resolve=True,
    )

    if not isinstance(dataset_cfg, dict):
        raise TypeError("cfg.data.dataset 必须能解析为 dict。")

    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )

    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    # YAML 可额外指定 metadata_keys；这里与默认离散标签集合取并集。
    configured_metadata_keys = set(
        cfg_select(cfg, "data.metadata_keys", [])
    )
    metadata_keys = DEFAULT_METADATA_KEYS | configured_metadata_keys

    # 与原 train.py 类似，对 action 等连续变量做 normalizer。
    # 但 phase_id / episode_id 等离散标签绝不能归一化。
    for column in cfg.data.dataset.keys_to_load:
        if column.startswith("pixels"):
            continue

        if column in metadata_keys:
            print(f"[OOT data] metadata 不归一化：{column}")
            continue

        try:
            normalizer = get_column_normalizer(
                dataset,
                column,
                column,
            )
            transforms.append(normalizer)
            print(f"[OOT data] 连续变量归一化：{column}")
        except Exception as exc:
            # 第一版对不支持 normalizer 的附加标签采取保守策略：
            # 保留原值，同时打印原因。
            print(
                f"[OOT data][WARN] 列 {column!r} 未归一化："
                f"{type(exc).__name__}: {exc}"
            )

    # action encoder 的原始输入维度应与数据中的 action 维度一致。
    # frameskip>1 时，stable-worldmodel 可能把多步 action 拼接起来。
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip
            * dataset.get_dim("action")
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    random_generator = torch.Generator().manual_seed(int(cfg.seed))

    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1.0 - cfg.train_split],
        generator=random_generator,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=random_generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    # ==================================================================
    # 2. Model and optimizer
    # ==================================================================
    world_model = hydra.utils.instantiate(cfg.model)

    if not hasattr(world_model, "remember"):
        raise TypeError(
            "当前 cfg.model 没有 remember()。"
            "运行 train_oot.py 时应使用 OOT 模型配置，例如 model=oot_leaky。"
        )

    scheduler_type = cfg_select(
        cfg,
        "scheduler.type",
        "LinearWarmupCosineAnnealingLR",
    )

    scheduler_interval = cfg_select(
        cfg,
        "scheduler.interval",
        "epoch",
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": scheduler_type},
            "interval": scheduler_interval,
        }
    }

    data_module = spt.data.DataModule(
        train=train_loader,
        val=val_loader,
    )

    lightning_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(oot_forward, cfg=cfg),
        optim=optimizers,
    )

    # ==================================================================
    # 3. Output directory and logger
    # ==================================================================
    run_id = cfg.get("subdir") or ""

    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
        run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        OmegaConf.save(cfg, file)

    logger = None

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    callbacks: list[pl.Callback] = [
        SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg.model,
            epoch_interval=1,
        )
    ]

    # 可通过命令行临时开启：
    #   ++analysis.enabled=true
    #   ++analysis.save_dir=/root/.../oot_latents
    if bool(cfg_select(cfg, "analysis.enabled", False)):
        analysis_save_dir = Path(
            str(
                cfg_select(
                    cfg,
                    "analysis.save_dir",
                    run_dir / "oot_analysis",
                )
            )
        )

        callbacks.append(
            OOTLatentDumpCallback(
                save_dir=analysis_save_dir,
                every_n_epochs=int(
                    cfg_select(
                        cfg,
                        "analysis.every_n_epochs",
                        1,
                    )
                ),
                max_batches_per_epoch=int(
                    cfg_select(
                        cfg,
                        "analysis.max_batches_per_epoch",
                        4,
                    )
                ),
                max_samples_per_batch=int(
                    cfg_select(
                        cfg,
                        "analysis.max_samples_per_batch",
                        8,
                    )
                ),
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"

    manager = spt.Manager(
        trainer=trainer,
        module=lightning_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()


if __name__ == "__main__":
    run()
