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

4. 严格区分两类 action 时间语义：
       - previous_action[t] = a_(t-1)，用于把记忆更新到状态 s_t；
       - current_action[t]  = a_t，用于从状态 s_t 预测 s_(t+1)。

5. 可选地把少量 validation latent 保存为 .npz，
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
import h5py
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
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
    "frame_loss_weight",
    "reset_mask",
}


def cfg_select(cfg: DictConfig, path: str, default: Any) -> Any:
    """
    安全读取可选 Hydra 配置。

    例如：
        cfg_select(cfg, "oot_loss.reappearance_weight", 1.0)

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


def build_previous_action_embedding(
    current_action_emb: Tensor,
    reset_mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    把用于“预测下一帧”的当前 action，右移成用于“更新当前记忆”的前一 action。

    OOT V3 的数据时间语义是：

        frame t 展示状态 s_t
        action[t] = a_t 负责产生 s_(t+1)

    因而，对第 t 个视觉 embedding z_t：

        previous_action[t] = a_(t-1)

    才是已经作用并形成当前状态 s_t 的动作；而 a_t 尚未作用，它应该留给
    predictor 去预测下一状态。

    对 episode 第一帧不存在 a_(-1)，因此使用全零 action embedding 作为
    “无前序动作”的中性标记。若 reset_mask 在序列中间为 1，也必须把对应
    时刻的 previous action 清零，避免上一 episode 的最后动作污染新 episode。

    Parameters
    ----------
    current_action_emb:
        shape = (B,T,D_action)
        第 t 项是 a_t 的 action embedding。

    reset_mask:
        shape = (B,T) 或 (B,T,1)
        1 表示当前帧是新 episode 的起点。

    Returns
    -------
    previous_action_emb:
        shape = (B,T,D_action)

        previous_action_emb[:, 0] = 0
        previous_action_emb[:, t] = current_action_emb[:, t-1], t >= 1
    """
    if current_action_emb.ndim != 3:
        raise ValueError(
            "current_action_emb 应为 (B,T,D_action)，"
            f"实际 shape={tuple(current_action_emb.shape)}。"
        )

    # t=0 没有前一动作；其余时刻把 action embedding 整体右移一格。
    no_previous_action = torch.zeros_like(current_action_emb[:, :1])
    previous_action_emb = torch.cat(
        [no_previous_action, current_action_emb[:, :-1]],
        dim=1,
    )

    if reset_mask is None:
        return previous_action_emb

    reset_mask = squeeze_last_singleton(reset_mask)

    expected_shape = current_action_emb.shape[:2]
    if tuple(reset_mask.shape) != tuple(expected_shape):
        raise ValueError(
            "reset_mask 应与 action 的 batch/time 维一致："
            f"expected={tuple(expected_shape)}, actual={tuple(reset_mask.shape)}。"
        )

    reset_mask = reset_mask.to(
        device=current_action_emb.device,
        dtype=current_action_emb.dtype,
    ).unsqueeze(-1)

    # reset 时刻不能继承 reset 前的 action。
    return previous_action_emb * (1.0 - reset_mask)


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


def safe_weighted_mean(
    values: Tensor,
    weights: Tensor,
) -> Tensor:
    """
    按权重求均值：

        sum(values * weights) / sum(weights)

    与先乘权重、再除以有效元素数量不同，这种写法不会因为
    提高少数时间步的权重而无意中整体放大 loss 尺度。

    若当前 batch 的权重总和为 0，则返回 0，而不是 NaN。
    """
    weights = weights.to(
        device=values.device,
        dtype=values.dtype,
    )

    numerator = (values * weights).sum()
    denominator = weights.sum().clamp_min(1.0)

    return numerator / denominator


def conditioned_centered_features(
    features: Tensor,
    sample_mask: Tensor,
    condition_labels: Tensor,
    auxiliary_labels: Optional[Tensor] = None,
    min_samples_per_condition: int = 2,
) -> tuple[Tensor, Optional[Tensor], Tensor, int]:
    """
    在每个 condition 内中心化，再把残差拼接起来。

    OOT 的固定流程使绝对 timestep、phase 和 visibility 高度相关。如果直接
    把 (B,T,D) 展平，模型只需为每个时刻输出一个共享模板，就能制造很大的
    全局方差。这里在相同 target timestep 的不同 episode 之间做中心化：

        residual[b,t] = z[b,t] - mean_b(z[b,t])

    因而任何只依赖帧数/phase 的公共信号都会被严格消掉，剩下的只能是
    episode、内容、位置或运动状态之间的差异。
    """
    if features.ndim != 3:
        raise ValueError(
            "conditioned_centered_features 需要 (B,T,D)，"
            f"实际收到 {tuple(features.shape)}。"
        )

    expected_shape = features.shape[:2]
    sample_mask = squeeze_last_singleton(sample_mask)
    condition_labels = squeeze_last_singleton(condition_labels)

    if tuple(sample_mask.shape) != tuple(expected_shape):
        raise ValueError(
            "sample_mask 与 features 的 batch/time 维不一致："
            f"mask={tuple(sample_mask.shape)}, "
            f"features={tuple(features.shape)}。"
        )

    if tuple(condition_labels.shape) != tuple(expected_shape):
        raise ValueError(
            "condition_labels 与 features 的 batch/time 维不一致："
            f"labels={tuple(condition_labels.shape)}, "
            f"features={tuple(features.shape)}。"
        )

    if auxiliary_labels is not None:
        auxiliary_labels = squeeze_last_singleton(auxiliary_labels)
        if tuple(auxiliary_labels.shape) != tuple(expected_shape):
            raise ValueError(
                "auxiliary_labels 与 features 的 batch/time 维不一致。"
            )

    valid = sample_mask > 0.5
    centered_chunks = []
    auxiliary_chunks = []
    centered_condition_chunks = []
    condition_count = 0

    # timestep 是离散整型；只遍历当前 batch 实际出现且有效的取值。
    active_conditions = torch.unique(condition_labels[valid])

    for condition in active_conditions:
        current_mask = valid & (condition_labels == condition)
        current_count = int(current_mask.sum().item())

        if current_count < int(min_samples_per_condition):
            continue

        current_features = features[current_mask]
        centered_chunks.append(
            current_features
            - current_features.mean(dim=0, keepdim=True)
        )
        centered_condition_chunks.append(
            condition_labels[current_mask]
        )

        if auxiliary_labels is not None:
            auxiliary_chunks.append(auxiliary_labels[current_mask])

        condition_count += 1

    if not centered_chunks:
        empty_features = features.new_empty((0, features.size(-1)))
        empty_auxiliary = None
        if auxiliary_labels is not None:
            empty_auxiliary = auxiliary_labels.new_empty((0,))
        empty_conditions = condition_labels.new_empty((0,))
        return empty_features, empty_auxiliary, empty_conditions, 0

    centered = torch.cat(centered_chunks, dim=0)
    centered_auxiliary = None
    if auxiliary_labels is not None:
        centered_auxiliary = torch.cat(auxiliary_chunks, dim=0)
    centered_conditions = torch.cat(centered_condition_chunks, dim=0)

    return (
        centered,
        centered_auxiliary,
        centered_conditions,
        condition_count,
    )


def vicreg_variance_covariance(
    centered_features: Tensor,
    target_std: float,
    eps: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """对已按 timestep 中心化的特征计算 VICReg variance/covariance。"""
    if centered_features.ndim != 2:
        raise ValueError(
            "vicreg_variance_covariance 需要 (N,D)，"
            f"实际收到 {tuple(centered_features.shape)}。"
        )

    if centered_features.size(0) < 2:
        zero = centered_features.new_zeros(())
        return zero, zero, zero

    features = centered_features.float()
    feature_std = torch.sqrt(
        features.var(dim=0, unbiased=False) + float(eps)
    )
    variance_loss = F.relu(float(target_std) - feature_std).mean()

    covariance = (
        features.T @ features
        / float(max(features.size(0) - 1, 1))
    )
    covariance_off_diagonal = covariance - torch.diag_embed(
        torch.diagonal(covariance)
    )
    covariance_loss = (
        covariance_off_diagonal.square().sum()
        / float(features.size(-1))
    )

    return variance_loss, covariance_loss, feature_std.mean()


def supervised_contrastive_loss(
    features: Tensor,
    labels: Optional[Tensor],
    temperature: float,
    condition_labels: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    用 content_id 约束同内容表示接近、不同内容表示分离。

    features 已先在相同 timestep 内跨 episode 中心化。对比也只在
    相同 timestep 内进行：相同 content 是正样本，不同 content 是负样本。
    这避免把同一 content 的不同运动时刻强行拉成一个静态原型。

    返回 (loss, valid_anchor_fraction)。只有同时拥有正样本和负样本
    的 anchor 才参与 loss；否则返回 0，避免“全是正样本”鼓励坍塌。
    """
    if labels is None or features.size(0) < 2:
        zero = features.new_zeros(())
        return zero, zero

    if float(temperature) <= 0.0:
        raise ValueError("content contrastive temperature 必须 > 0。")

    labels = labels.reshape(-1)
    if labels.numel() != features.size(0):
        raise ValueError("contrastive labels 数量与 features 不一致。")

    if condition_labels is not None:
        condition_labels = condition_labels.reshape(-1)
        if condition_labels.numel() != features.size(0):
            raise ValueError(
                "contrastive condition labels 数量与 features 不一致。"
            )

    normalized = F.normalize(features.float(), dim=-1)
    logits = normalized @ normalized.T
    logits = logits / float(temperature)

    sample_count = logits.size(0)
    diagonal = torch.eye(
        sample_count,
        device=logits.device,
        dtype=torch.bool,
    )
    comparison_mask = ~diagonal
    if condition_labels is not None:
        comparison_mask = comparison_mask & (
            condition_labels[:, None] == condition_labels[None, :]
        )

    same_content = labels[:, None] == labels[None, :]
    positive_mask = comparison_mask & same_content
    negative_mask = comparison_mask & ~same_content

    valid_anchor = (
        positive_mask.any(dim=1)
        & negative_mask.any(dim=1)
    )
    valid_anchor_fraction = valid_anchor.float().mean()
    if not bool(valid_anchor.any()):
        return features.new_zeros(()), valid_anchor_fraction

    anchor_logits = logits[valid_anchor]
    anchor_comparison_mask = comparison_mask[valid_anchor]
    anchor_positive_mask = positive_mask[valid_anchor]
    anchor_logits = anchor_logits.masked_fill(
        ~anchor_comparison_mask,
        float("-inf"),
    )
    log_prob = anchor_logits - torch.logsumexp(
        anchor_logits,
        dim=1,
        keepdim=True,
    )
    positive_count = anchor_positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_prob = (
        log_prob.masked_fill(~anchor_positive_mask, 0.0).sum(dim=1)
        / positive_count
    )

    return -mean_positive_log_prob.mean(), valid_anchor_fraction


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

    # ---------- 最终再现与首次再现 ----------
    # OOT V3 中每条 episode 有两次再现：
    #   phase 2：第一次再现
    #   phase 4：最终再现
    #
    # is_final_reappearance 是 is_reappearance 的子集，因此：
    #
    #   first_reappearance
    #       = reappearance - final_reappearance
    #
    final_reappearance_raw = first_existing(
        batch,
        ["is_final_reappearance"],
    )
    final_reappearance_target = align_target_time(
        final_reappearance_raw,
        num_preds,
        context_length,
    )
    final_reappearance_mask = as_float_mask(
        final_reappearance_target,
        step_mse,
    )

    if final_reappearance_mask is not None:
        final_reappearance_mask = (
            final_reappearance_mask * valid_mask
        )

    first_reappearance_mask = None

    if (
        reappearance_mask is not None
        and final_reappearance_mask is not None
    ):
        first_reappearance_mask = (
            reappearance_mask - final_reappearance_mask
        ).clamp_min(0.0)

    # 普通可见帧：对象可见，但不是刚刚解除遮挡的再现帧。
    #
    # visible 本身仍保留为宽泛指标；
    # visible_non_reappearance 用于衡量稳定可见阶段的预测误差。
    visible_non_reappearance_mask = None

    if (
        visible_mask is not None
        and reappearance_mask is not None
    ):
        visible_non_reappearance_mask = (
            visible_mask - reappearance_mask
        ).clamp_min(0.0)

    return {
        "valid": valid_mask,
        "visible": visible_mask,
        "visible_non_reappearance": visible_non_reappearance_mask,
        "occluded": occluded_mask,
        "reappearance": reappearance_mask,
        "first_reappearance": first_reappearance_mask,
        "final_reappearance": final_reappearance_mask,
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
    configured_sigreg_weight = float(cfg.loss.sigreg.weight)
    configured_variance_weight = float(
        cfg_select(cfg, "oot_loss.variance_weight", 1.0)
    )
    variance_target_std = float(
        cfg_select(cfg, "oot_loss.variance_target_std", 1.0)
    )
    regularizer_mode = str(
        cfg_select(
            cfg,
            "oot_loss.regularizer_mode",
            "legacy_global",
        )
    )

    if regularizer_mode not in {
        "legacy_global",
        "conditioned_vicreg_v2",
    }:
        raise ValueError(
            "oot_loss.regularizer_mode 必须是 "
            "legacy_global 或 conditioned_vicreg_v2，"
            f"实际收到 {regularizer_mode!r}。"
        )

    use_conditioned_v2 = regularizer_mode == "conditioned_vicreg_v2"

    # legacy_global 保持旧实验完全兼容；v2 默认关闭会被 phase/时间模板钻空子
    # 的全局 SIGReg 与全局 variance floor。
    global_sigreg_weight = float(
        cfg_select(
            cfg,
            "oot_loss.global_sigreg_weight",
            0.0 if use_conditioned_v2 else configured_sigreg_weight,
        )
    )
    legacy_variance_weight = float(
        cfg_select(
            cfg,
            "oot_loss.legacy_global_variance_weight",
            0.0 if use_conditioned_v2 else configured_variance_weight,
        )
    )
    stop_gradient_target = bool(
        cfg_select(
            cfg,
            "oot_loss.stop_gradient_target",
            use_conditioned_v2,
        )
    )
    conditioned_variance_weight = float(
        cfg_select(
            cfg,
            "oot_loss.conditioned_variance_weight",
            25.0,
        )
    )
    conditioned_covariance_weight = float(
        cfg_select(
            cfg,
            "oot_loss.conditioned_covariance_weight",
            1.0,
        )
    )
    content_contrastive_weight = float(
        cfg_select(
            cfg,
            "oot_loss.content_contrastive_weight",
            1.0,
        )
    )
    content_temperature = float(
        cfg_select(
            cfg,
            "oot_loss.content_temperature",
            0.1,
        )
    )
    regularize_prediction = bool(
        cfg_select(
            cfg,
            "oot_loss.regularize_prediction",
            True,
        )
    )
    min_samples_per_timestep = int(
        cfg_select(
            cfg,
            "oot_loss.min_samples_per_timestep",
            2,
        )
    )

    if content_temperature <= 0.0:
        raise ValueError("oot_loss.content_temperature 必须 > 0。")
    if min_samples_per_timestep < 2:
        raise ValueError(
            "oot_loss.min_samples_per_timestep 必须 >= 2。"
        )

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

    # predictor 的第 t 个位置使用状态 s_t 预测 s_(t+1)。
    # current_action[t] = a_t 正是 s_t -> s_(t+1) 的控制命令。
    context_emb = emb[:, :context_length]
    current_action = act_emb[:, :context_length]
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

    # ------------------------------------------------------------------
    # Action 时间对齐
    # ------------------------------------------------------------------
    # 当前 frame t 是此前 action[t-1] 作用后的结果，因此 Leaky-RNN 在读取
    # z_t 时应接收 previous_action[t] = action[t-1]。
    #
    # 与此相对，predictor 预测 z_(t+1) 时应接收 current_action[t] = action[t]。
    # 这样可以把“状态估计”和“状态转移”清楚分开。
    previous_action = build_previous_action_embedding(
        current_action,
        reset_mask=input_reset_mask,
    )

    memory_emb, memory_hidden, final_memory = self.model.remember(
        context_emb,
        previous_action,
        valid_mask=input_valid_mask,
        reset_mask=input_reset_mask,
    )

    # ------------------------------------------------------------------
    # 3. predict：用当前状态记忆 h_t 与当前控制 a_t 预测 z_(t+1)
    # ------------------------------------------------------------------
    pred_emb = self.model.predict(memory_emb, current_action)

    if pred_emb.shape != target_emb.shape:
        raise RuntimeError(
            "预测与 target 的形状不一致："
            f"pred={tuple(pred_emb.shape)}, target={tuple(target_emb.shape)}。"
        )

    # v2 对 prediction target 使用 stop-gradient，防止 encoder 与 predictor
    # 共同移动到“只编码 timestep/phase”的容易预测目标。target_emb 本身仍保留
    # 梯度，供下方 conditioned variance/covariance/content loss 训练 encoder。
    prediction_target = (
        target_emb.detach()
        if stop_gradient_target
        else target_emb
    )

    # 每个 batch、每个预测时刻独立计算 latent MSE。
    # 对 embedding dimension 求均值后得到 (B,T_context)。
    step_mse = (
        pred_emb - prediction_target
    ).pow(2).mean(dim=-1)

    masks = get_sequence_masks(
        clean_batch,
        num_preds,
        context_length,
        step_mse,
        cfg,
    )

    all_pred_loss = safe_masked_mean(step_mse, masks["valid"])

    # --------------------------------------------------------------
    # Prediction loss weights
    #
    # Primitive v1 在 HDF5 中直接保存逐帧 frame_loss_weight：
    #
    #   phase 0: [1, 1, 1]
    #   phase 1: [0.1, 0.1, 0.1, 0.1]
    #   phase 2: [3, 2, 2]
    #   phase 3: [0.1 x 8]
    #   phase 4: [3, 2, 2, 2]
    #
    # predictor 的第 t 项预测目标 frame t+1，因此权重也必须像
    # valid_mask / phase_id 一样对齐到 target 时刻，而不能直接使用
    # 输入 frame t 的权重。
    #
    # 对不含 frame_loss_weight 的旧 OOT-v3 数据，仍回退到原有的
    # reappearance_weight 逻辑，从而保持 Epoch24 实验兼容性。
    # --------------------------------------------------------------
    frame_loss_weight_raw = first_existing(
        clean_batch,
        ["frame_loss_weight"],
    )

    use_frame_loss_weight = bool(
        cfg_select(
            cfg,
            "oot_loss.use_frame_loss_weight",
            frame_loss_weight_raw is not None,
        )
    )

    if use_frame_loss_weight:
        if frame_loss_weight_raw is None:
            raise KeyError(
                "oot_loss.use_frame_loss_weight=true，"
                "但 batch 中没有 frame_loss_weight。"
            )

        frame_loss_weight_target = align_target_time(
            frame_loss_weight_raw,
            num_preds,
            context_length,
        )

        frame_loss_weight_target = squeeze_last_singleton(
            frame_loss_weight_target
        )

        if frame_loss_weight_target.shape != step_mse.shape:
            raise RuntimeError(
                "frame_loss_weight 与预测 target 未正确对齐："
                f"weight={tuple(frame_loss_weight_target.shape)}, "
                f"step_mse={tuple(step_mse.shape)}。"
            )

        frame_loss_weight_target = torch.nan_to_num(
            frame_loss_weight_target,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(
            device=step_mse.device,
            dtype=step_mse.dtype,
        ).clamp_min(0.0)

        prediction_weights = (
            masks["valid"]
            * frame_loss_weight_target
        )

    else:
        # 旧 OOT-v3 的兼容分支。
        reappearance_weight = float(
            cfg_select(
                cfg,
                "oot_loss.reappearance_weight",
                1.0,
            )
        )

        prediction_weights = masks["valid"].clone()

        if masks["reappearance"] is not None:
            prediction_weights = prediction_weights + (
                reappearance_weight - 1.0
            ) * masks["reappearance"]

    weighted_pred_loss = safe_weighted_mean(
        step_mse,
        prediction_weights,
    )

    # ------------------------------------------------------------------
    # Anti-collapse regularization
    # ------------------------------------------------------------------
    # legacy_global 仅用于复现旧实验。
    #
    # conditioned_vicreg_v2 的关键区别：
    #   1. 只使用 visible/reappearance target；遮挡空白帧不参与；
    #   2. 在相同 target timestep 内跨 episode 中心化；
    #   3. 对中心化残差施加 variance + covariance；
    #   4. 在相同 timestep 内用 content_id 做监督式对比，迫使
    #      残差携带对象内容，但不抹掉跨时刻的状态变化。
    #
    # 因此任何对所有 episode 都相同的“第 t 帧模板”在进入 loss 前都会被
    # 减成 0，无法再冒充健康方差。
    with torch.autocast(
        device_type=emb.device.type,
        enabled=False,
    ):
        # 保留 legacy 指标用于与历史日志直接比较，但 v2 默认权重为 0。
        sigreg_features = emb.float().reshape(
            1,
            -1,
            emb.size(-1),
        )
        sigreg_loss = self.sigreg(sigreg_features)

        flat_features = sigreg_features.squeeze(0)
        feature_std = torch.sqrt(
            flat_features.var(
                dim=0,
                unbiased=False,
            ) + 1.0e-4
        )

        encoder_feature_std = feature_std.mean()

        legacy_global_variance_loss = F.relu(
            variance_target_std - feature_std
        ).mean()

        conditioned_variance_loss = emb.new_zeros((), dtype=torch.float32)
        conditioned_covariance_loss = emb.new_zeros((), dtype=torch.float32)
        conditioned_encoder_variance_loss = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_encoder_covariance_loss = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_prediction_variance_loss = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_prediction_covariance_loss = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_encoder_feature_std = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_prediction_feature_std = emb.new_zeros(
            (), dtype=torch.float32
        )
        content_contrastive_loss = emb.new_zeros((), dtype=torch.float32)
        content_contrastive_valid_anchor_fraction = emb.new_zeros(
            (), dtype=torch.float32
        )
        conditioned_sample_count = emb.new_zeros((), dtype=torch.float32)
        conditioned_timestep_count = emb.new_zeros((), dtype=torch.float32)

        if use_conditioned_v2:
            visible_mask = masks.get("visible")
            if visible_mask is None:
                raise KeyError(
                    "conditioned_vicreg_v2 需要 object_visible/is_visible。"
                )

            target_timestep = align_target_time(
                first_existing(clean_batch, ["timestep"]),
                num_preds,
                context_length,
            )
            if target_timestep is None:
                raise KeyError(
                    "conditioned_vicreg_v2 需要 batch['timestep']。"
                )

            target_content_id = align_target_time(
                first_existing(clean_batch, ["content_id"]),
                num_preds,
                context_length,
            )
            if (
                content_contrastive_weight > 0.0
                and target_content_id is None
            ):
                raise KeyError(
                    "content_contrastive_weight>0 需要 batch['content_id']。"
                )

            (
                encoder_centered,
                encoder_content_labels,
                encoder_timestep_labels,
                timestep_count,
            ) = (
                conditioned_centered_features(
                    target_emb.float(),
                    sample_mask=visible_mask,
                    condition_labels=target_timestep,
                    auxiliary_labels=target_content_id,
                    min_samples_per_condition=min_samples_per_timestep,
                )
            )

            (
                conditioned_encoder_variance_loss,
                conditioned_encoder_covariance_loss,
                conditioned_encoder_feature_std,
            ) = vicreg_variance_covariance(
                encoder_centered,
                target_std=variance_target_std,
            )

            contrastive_features = encoder_centered
            contrastive_labels = encoder_content_labels
            contrastive_timestep_labels = encoder_timestep_labels

            if regularize_prediction:
                (
                    prediction_centered,
                    prediction_content_labels,
                    prediction_timestep_labels,
                    prediction_timestep_count,
                ) = conditioned_centered_features(
                    pred_emb.float(),
                    sample_mask=visible_mask,
                    condition_labels=target_timestep,
                    auxiliary_labels=target_content_id,
                    min_samples_per_condition=min_samples_per_timestep,
                )

                if prediction_timestep_count != timestep_count:
                    raise RuntimeError(
                        "encoder/prediction conditioned timestep 数量不一致。"
                    )

                (
                    conditioned_prediction_variance_loss,
                    conditioned_prediction_covariance_loss,
                    conditioned_prediction_feature_std,
                ) = vicreg_variance_covariance(
                    prediction_centered,
                    target_std=variance_target_std,
                )

                conditioned_variance_loss = 0.5 * (
                    conditioned_encoder_variance_loss
                    + conditioned_prediction_variance_loss
                )
                conditioned_covariance_loss = 0.5 * (
                    conditioned_encoder_covariance_loss
                    + conditioned_prediction_covariance_loss
                )

                if encoder_content_labels is not None:
                    contrastive_features = torch.cat(
                        [encoder_centered, prediction_centered],
                        dim=0,
                    )
                    contrastive_labels = torch.cat(
                        [
                            encoder_content_labels,
                            prediction_content_labels,
                        ],
                        dim=0,
                    )
                    contrastive_timestep_labels = torch.cat(
                        [
                            encoder_timestep_labels,
                            prediction_timestep_labels,
                        ],
                        dim=0,
                    )
            else:
                conditioned_variance_loss = (
                    conditioned_encoder_variance_loss
                )
                conditioned_covariance_loss = (
                    conditioned_encoder_covariance_loss
                )

            (
                content_contrastive_loss,
                content_contrastive_valid_anchor_fraction,
            ) = supervised_contrastive_loss(
                contrastive_features,
                contrastive_labels,
                temperature=content_temperature,
                condition_labels=contrastive_timestep_labels,
            )
            conditioned_sample_count = emb.new_tensor(
                float(encoder_centered.size(0)),
                dtype=torch.float32,
            )
            conditioned_timestep_count = emb.new_tensor(
                float(timestep_count),
                dtype=torch.float32,
            )

    # 不把权重为 0 的项写成 0 * loss：如果仅用于诊断的 legacy
    # loss 意外产生 NaN，IEEE 浮点下 0 * NaN 仍是 NaN，会污染总 loss。
    total_loss = weighted_pred_loss
    if global_sigreg_weight != 0.0:
        total_loss = total_loss + global_sigreg_weight * sigreg_loss
    if legacy_variance_weight != 0.0:
        total_loss = (
            total_loss
            + legacy_variance_weight * legacy_global_variance_loss
        )
    if conditioned_variance_weight != 0.0:
        total_loss = (
            total_loss
            + conditioned_variance_weight * conditioned_variance_loss
        )
    if conditioned_covariance_weight != 0.0:
        total_loss = (
            total_loss
            + conditioned_covariance_weight * conditioned_covariance_loss
        )
    if content_contrastive_weight != 0.0:
        total_loss = (
            total_loss
            + content_contrastive_weight * content_contrastive_loss
        )

    # ------------------------------------------------------------------
    # 4. 诊断指标
    # ------------------------------------------------------------------
    # cosine error 只作为指标，不代替原始 MSE。
    cosine_error_per_step = 1.0 - F.cosine_similarity(
        pred_emb,
        prediction_target,
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
        # variance_loss 保留兼容名称：v2 时表示 conditioned variance，
        # legacy 时表示旧的全局 variance floor。
        "variance_loss": (
            conditioned_variance_loss
            if use_conditioned_v2
            else legacy_global_variance_loss
        ),
        "legacy_global_variance_loss": legacy_global_variance_loss,
        "conditioned_variance_loss": conditioned_variance_loss,
        "conditioned_covariance_loss": conditioned_covariance_loss,
        "conditioned_encoder_variance_loss": (
            conditioned_encoder_variance_loss
        ),
        "conditioned_encoder_covariance_loss": (
            conditioned_encoder_covariance_loss
        ),
        "conditioned_prediction_variance_loss": (
            conditioned_prediction_variance_loss
        ),
        "conditioned_prediction_covariance_loss": (
            conditioned_prediction_covariance_loss
        ),
        "content_contrastive_loss": content_contrastive_loss,
        "content_contrastive_valid_anchor_fraction": (
            content_contrastive_valid_anchor_fraction
        ),
        "conditioned_encoder_feature_std": (
            conditioned_encoder_feature_std
        ),
        "conditioned_prediction_feature_std": (
            conditioned_prediction_feature_std
        ),
        "conditioned_sample_count": conditioned_sample_count,
        "conditioned_timestep_count": conditioned_timestep_count,
        "encoder_feature_std": encoder_feature_std,
        "cosine_error": cosine_error,
        "memory_hidden_norm": memory_hidden_norm,
        "memory_step_change": memory_step_change,
        "leak_rate": self.model.leak_rate,
        "initial_state_std": memory_hidden.new_tensor(
            float(self.model.memory.initial_state_std)
        ),
        "stop_gradient_target": memory_hidden.new_tensor(
            float(stop_gradient_target)
        ),
        "global_sigreg_weight": memory_hidden.new_tensor(
            global_sigreg_weight
        ),
        "legacy_variance_weight": memory_hidden.new_tensor(
            legacy_variance_weight
        ),
    }


    # ------------------------------------------------------------------
    # Representation-scale diagnostics
    # ------------------------------------------------------------------
    # 这些指标不参与反向传播，只用于识别：
    # - target / prediction 是否一起缩向零；
    # - memory hidden 正常但 downstream embedding 是否坍缩；
    # - 小 MSE 是否只是由小尺度造成。
    def diagnostic_rms(x: Tensor) -> Tensor:
        x32 = x.detach().float()
        return x32.square().mean().sqrt()

    def diagnostic_std(x: Tensor) -> Tensor:
        return x.detach().float().std(
            unbiased=False
        )

    target_rms = diagnostic_rms(target_emb)
    pred_rms = diagnostic_rms(pred_emb)

    output.update(
        {
            "emb_rms": diagnostic_rms(emb),
            "emb_std": diagnostic_std(emb),
            "context_emb_rms": diagnostic_rms(
                context_emb
            ),
            "target_emb_rms": target_rms,
            "target_emb_std": diagnostic_std(
                target_emb
            ),
            "pred_emb_rms": pred_rms,
            "pred_emb_std": diagnostic_std(
                pred_emb
            ),
            "memory_emb_rms": diagnostic_rms(
                memory_emb
            ),
            "memory_emb_std": diagnostic_std(
                memory_emb
            ),
            "memory_hidden_rms": diagnostic_rms(
                memory_hidden
            ),
            "pred_target_rms_ratio": (
                pred_rms
                / target_rms.clamp_min(1e-12)
            ),
        }
    )

    # 只有相应标签存在时才记录分阶段 MSE。
    if masks["visible"] is not None:
        output["pred_loss_visible"] = safe_masked_mean(
            step_mse,
            masks["visible"],
        )

    if masks["visible_non_reappearance"] is not None:
        output[
            "pred_loss_visible_non_reappearance"
        ] = safe_masked_mean(
            step_mse,
            masks["visible_non_reappearance"],
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

    if masks["first_reappearance"] is not None:
        output["pred_loss_first_reappearance"] = safe_masked_mean(
            step_mse,
            masks["first_reappearance"],
        )

    if masks["final_reappearance"] is not None:
        output["pred_loss_final_reappearance"] = safe_masked_mean(
            step_mse,
            masks["final_reappearance"],
        )

    # --------------------------------------------------------------
    # phase 0–4 的分阶段预测误差
    #
    # 这里只增加诊断指标，不参与训练优化目标。
    # 默认统计 OOT V3 的五个 phase；也可在配置中通过
    # oot_loss.phase_ids 覆盖。
    # --------------------------------------------------------------
    if masks["phase_id"] is not None:
        phase_ids = cfg_select(
            cfg,
            "oot_loss.phase_ids",
            [0, 1, 2, 3, 4],
        )

        for phase_id in phase_ids:
            phase_mask = values_to_mask(
                masks["phase_id"],
                [int(phase_id)],
                step_mse,
            )

            if phase_mask is not None:
                phase_mask = phase_mask * masks["valid"]

                output[
                    f"pred_loss_phase_{int(phase_id)}"
                ] = safe_masked_mean(
                    step_mse,
                    phase_mask,
                )

    # --------------------------------------------------------------
    # 按实际参与该指标计算的元素数量聚合 epoch 指标
    #
    # 不能让所有指标统一使用 emb.size(0)：
    # 不同 batch 中各 phase / visibility 类别包含的时间步数量可能不同。
    # 若统一按 episode 数加权，epoch 结果就不是严格的逐帧总体均值。
    # --------------------------------------------------------------
    def mask_element_count(
        mask: Optional[Tensor],
    ) -> int:
        if mask is None:
            return 0

        return int(
            mask.detach().sum().item()
        )

    default_batch_weight = int(emb.size(0))
    valid_count = mask_element_count(masks["valid"])

    # pred_loss 使用 prediction_weights 作为分母，
    # 因此跨 batch 聚合时也应按权重总和加权。
    prediction_weight_sum = float(
        prediction_weights.detach().sum().item()
    )

    metric_log_weights = {
        "loss": default_batch_weight,
        "pred_loss": prediction_weight_sum,
        "pred_loss_all": valid_count,
        "sigreg_loss": default_batch_weight,
        "cosine_error": valid_count,
        "memory_hidden_norm": default_batch_weight,
        "memory_step_change": default_batch_weight,
        "leak_rate": default_batch_weight,

        "pred_loss_visible": mask_element_count(
            masks["visible"]
        ),
        "pred_loss_visible_non_reappearance": mask_element_count(
            masks["visible_non_reappearance"]
        ),
        "pred_loss_occluded": mask_element_count(
            masks["occluded"]
        ),
        "pred_loss_reappearance": mask_element_count(
            masks["reappearance"]
        ),
        "pred_loss_first_reappearance": mask_element_count(
            masks["first_reappearance"]
        ),
        "pred_loss_final_reappearance": mask_element_count(
            masks["final_reappearance"]
        ),
    }

    # phase 指标也按对应 phase 的实际预测帧数量加权。
    if masks["phase_id"] is not None:
        log_phase_ids = cfg_select(
            cfg,
            "oot_loss.phase_ids",
            [0, 1, 2, 3, 4],
        )

        for phase_id in log_phase_ids:
            phase_mask = values_to_mask(
                masks["phase_id"],
                [int(phase_id)],
                step_mse,
            )

            if phase_mask is not None:
                phase_mask = (
                    phase_mask * masks["valid"]
                )

                metric_log_weights[
                    f"pred_loss_phase_{int(phase_id)}"
                ] = mask_element_count(phase_mask)

    # 每个指标单独调用 self.log，才能为不同指标指定不同权重。
    # mask 元素数为 0 时不记录，避免把人为返回的 0 混入 epoch 平均。
    for name, value in output.items():
        if value.ndim != 0:
            continue

        log_weight = metric_log_weights.get(
            name,
            default_batch_weight,
        )

        if log_weight <= 0:
            continue

        self.log(
            f"{stage}/{name}",
            value.detach(),
            on_step=stage in {"fit", "train"},
            on_epoch=True,
            prog_bar=name in {"loss", "pred_loss"},
            sync_dist=True,
            batch_size=log_weight,
        )

    # ------------------------------------------------------------------
    # 5. 返回少量中间量，供 validation callback 保存
    # ------------------------------------------------------------------
    # 这些张量仍与计算图相连；callback 保存时会 detach。
    output.update(
        {
            "emb": emb,
            "context_emb": context_emb,
            "previous_action_emb": previous_action,
            "current_action_emb": current_action,
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

    for mask_name in [
        "valid",
        "visible",
        "visible_non_reappearance",
        "occluded",
        "reappearance",
        "first_reappearance",
        "final_reappearance",
    ]:
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




class RepresentationCollapseGuard(pl.Callback):
    """
    当 validation MSE 极小但 cosine error 很大时自动停止。

    这不是把小 MSE 直接判为坍缩；只有同时满足：
        pred_loss < mse_threshold
        cosine_error > cosine_threshold
    才累计一次可疑 epoch。
    """

    def __init__(
        self,
        mse_threshold: float = 1e-4,
        cosine_threshold: float = 0.8,
        patience: int = 1,
    ) -> None:
        super().__init__()

        self.mse_threshold = float(
            mse_threshold
        )
        self.cosine_threshold = float(
            cosine_threshold
        )
        self.patience = max(1, int(patience))
        self.bad_epochs = 0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module

        metrics = trainer.callback_metrics

        mse_value = metrics.get(
            "validate/pred_loss"
        )
        cosine_value = metrics.get(
            "validate/cosine_error"
        )

        if (
            mse_value is None
            or cosine_value is None
        ):
            return

        mse = float(
            mse_value.detach().cpu()
        )
        cosine = float(
            cosine_value.detach().cpu()
        )

        suspicious = (
            mse < self.mse_threshold
            and cosine > self.cosine_threshold
        )

        if suspicious:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        print("\n[representation collapse guard]")
        print("validation pred_loss:", mse)
        print("validation cosine_error:", cosine)
        print("suspicious:", suspicious)
        print(
            "consecutive suspicious epochs:",
            self.bad_epochs,
        )

        if self.bad_epochs >= self.patience:
            print(
                "STOP: probable representation-scale "
                "collapse detected"
            )
            trainer.should_stop = True



class StopAfterValidationEpochs(pl.Callback):
    """
    用于学习率扫描。

    Trainer 和 scheduler 仍按完整训练计划配置，
    但在指定次数的完整 validation 结束后主动停止。
    因此不会把 cosine schedule 压缩到一个 epoch。
    """

    def __init__(
        self,
        validation_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.validation_epochs = max(
            1,
            int(validation_epochs),
        )
        self.completed_validation_epochs = 0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module

        if trainer.sanity_checking:
            return

        self.completed_validation_epochs += 1

        print()
        print("[lr scan stop callback]")
        print(
            "completed validation epochs:",
            self.completed_validation_epochs,
        )
        print(
            "configured stop after:",
            self.validation_epochs,
        )

        if (
            self.completed_validation_epochs
            >= self.validation_epochs
        ):
            print(
                "LR_SCAN_STOP_AFTER_VALIDATION_EPOCHS"
            )
            trainer.should_stop = True


class FirstBackwardGradientAudit(pl.Callback):
    """Fail the run if any trainable model parameter misses the first backward."""

    def __init__(self) -> None:
        super().__init__()
        self.completed = False

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer
        if self.completed:
            return
        model = getattr(pl_module, "model", None)
        if model is None:
            raise RuntimeError("GRADIENT AUDIT FAILED: Lightning module has no model")
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        print("\n[first backward gradient audit]")
        print("trainable_parameters_without_gradient:", missing)
        if missing:
            raise RuntimeError(
                "GRADIENT AUDIT FAILED: trainable parameters without gradients: "
                f"{missing}"
            )
        self.completed = True


class FixedValidationRNGCallback(pl.Callback):

    """
    让每个 validation epoch 使用相同的随机初态序列。

    验证前保存 Python、NumPy、CPU Torch 和 CUDA RNG 状态；
    验证结束后恢复，因此不会让后续训练 epoch 重复使用同一组随机 h0。
    """

    def __init__(self, seed: int) -> None:
        super().__init__()

        self.seed = int(seed)

        self._python_state = None
        self._numpy_state = None
        self._torch_state = None
        self._cuda_states = None
        self._active = False

    def _restore_rng(self) -> None:
        if not self._active:
            return

        import random

        if self._python_state is not None:
            random.setstate(self._python_state)

        if self._numpy_state is not None:
            np.random.set_state(self._numpy_state)

        if self._torch_state is not None:
            torch.set_rng_state(self._torch_state)

        if (
            self._cuda_states is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                self._cuda_states
            )

        self._active = False

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module

        import random

        if self._active:
            raise RuntimeError(
                "FixedValidationRNGCallback "
                "在上一次 validation 后未恢复 RNG。"
            )

        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()

        if torch.cuda.is_available():
            self._cuda_states = (
                torch.cuda.get_rng_state_all()
            )
        else:
            self._cuda_states = None

        effective_seed = (
            self.seed
            + int(trainer.global_rank)
        )

        random.seed(effective_seed)

        np.random.seed(
            effective_seed % (2**32 - 1)
        )

        torch.manual_seed(effective_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                effective_seed
            )

        self._active = True

        print(
            "\n[validation RNG audit]"
        )
        print(
            "fixed validation seed:",
            effective_seed,
        )
        print(
            "training RNG will be restored: True"
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._restore_rng()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._restore_rng()


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
            "previous_action_emb",
            "current_action_emb",
            "memory_emb",
            "memory_hidden",
            "final_memory",
            "pred_emb",
            "target_emb",
            "step_mse",
            "target_phase_id",
            "target_valid_mask",
            "target_visible_mask",
            "target_visible_non_reappearance_mask",
            "target_occluded_mask",
            "target_reappearance_mask",
            "target_first_reappearance_mask",
            "target_final_reappearance_mask",
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

            # NumPy 不支持 PyTorch bfloat16；
            # analysis dump 统一转为 float32 后再保存。
            if detached.dtype == torch.bfloat16:
                detached = detached.float()

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
            "is_final_reappearance",
            "valid_mask",
            "episode_start",
            "object_x",
            "object_y",
            "object_position",
            "action",
            "object_angle",
            "object_orientation",
            "object_angular_velocity",
            "motion_type",
            "frame_loss_weight",
        ]

        for key in metadata_keys:
            value = batch.get(key)

            if not torch.is_tensor(value):
                continue

            detached = value.detach().cpu()

            if detached.ndim >= 1 and detached.shape[0] == batch["pixels"].shape[0]:
                detached = detached[:self.max_samples_per_batch]

            if detached.dtype == torch.bfloat16:
                detached = detached.float()

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

    # 必须在 dataset split、DataLoader 和模型实例化之前设 seed。
    seed = int(cfg.seed)
    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.benchmark = False

    print("\n[seed audit]")
    print("seed:", seed)
    print("seed set before dataset/model construction: True")

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

    # ---------------------------------------------------------
    # Local-only dataset resolution
    # ---------------------------------------------------------
    # 配置中通常写：
    #     oot_v3/train.h5
    #
    # 训练时将它显式解析成：
    #     $LOCAL_DATASET_DIR/oot_v3/train.h5
    #
    # 这样 stable_worldmodel 会收到本地绝对路径，
    # 而不会把 oot_v3/train.h5 当成 Hugging Face 文件名。
    local_dataset_root = os.environ.get(
        "LOCAL_DATASET_DIR"
    )

    local_only = (
        os.environ.get("LOCAL_ONLY_DATASET", "0")
        == "1"
    )

    cache_dir = None

    if local_dataset_root:
        local_dataset_root = Path(
            local_dataset_root
        ).expanduser().resolve()

        configured_path = Path(
            str(dataset_name)
        )

        if configured_path.is_absolute():
            dataset_path = configured_path
        else:
            dataset_path = (
                local_dataset_root
                / configured_path
            ).resolve()

        if dataset_path.is_file():
            dataset_name = str(dataset_path)
            cache_dir = None

            print("\n[dataset source]")
            print(
                "configured name: ",
                configured_path,
            )
            print(
                "resolved path:   ",
                dataset_path,
            )
            print(
                "exists:          ",
                dataset_path.is_file(),
            )
            print(
                "source:           LOCAL FILE ONLY"
            )

        elif local_only:
            raise FileNotFoundError(
                "LOCAL_ONLY_DATASET=1，但找不到本地数据文件："
                f"{dataset_path}"
            )

        else:
            # 保留非 OOT 数据集可能使用的原始 cache 行为。
            cache_dir = str(local_dataset_root)

    elif local_only:
        raise RuntimeError(
            "LOCAL_ONLY_DATASET=1，"
            "但 LOCAL_DATASET_DIR 没有设置。"
        )

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

    split_generator = torch.Generator().manual_seed(seed)
    train_loader_generator = torch.Generator().manual_seed(seed + 1)

    stratify_key = cfg_select(
        cfg,
        "split.stratify_key",
        None,
    )

    assignment_key = cfg_select(
        cfg,
        "split.assignment_key",
        None,
    )

    dataset_file_path = Path(str(dataset_name))

    if (
        assignment_key
        and dataset_file_path.is_file()
    ):
        # primitive_v2 writes a renderer/trajectory-grouped split_id into H5.
        # Using it here prevents matched-pair members or duplicated trajectories
        # from leaking across train, validation, and reserved test splits.
        train_value = int(
            cfg_select(cfg, "split.train_value", 0)
        )
        validation_value = int(
            cfg_select(cfg, "split.validation_value", 1)
        )

        with h5py.File(dataset_file_path, "r") as file:
            if assignment_key not in file:
                raise KeyError(
                    f"split.assignment_key={assignment_key!r}，"
                    "但 H5 中不存在该字段。"
                )
            episode_offsets = np.asarray(
                file["ep_offset"][:],
                dtype=np.int64,
            )
            assignments = np.asarray(
                file[assignment_key][episode_offsets]
            ).reshape(-1)

            trajectory_key = cfg_select(
                cfg,
                "split.trajectory_key",
                None,
            )
            if trajectory_key:
                if trajectory_key not in file:
                    raise KeyError(
                        f"split.trajectory_key={trajectory_key!r}，"
                        "但 H5 中不存在该字段。"
                    )
                trajectories = np.asarray(
                    file[trajectory_key][episode_offsets]
                ).reshape(-1)
                for trajectory in np.unique(trajectories):
                    trajectory_assignments = np.unique(
                        assignments[trajectories == trajectory]
                    )
                    if len(trajectory_assignments) != 1:
                        raise RuntimeError(
                            "同一 trajectory 出现在多个 split："
                            f"trajectory={int(trajectory)}，"
                            f"assignments={trajectory_assignments.tolist()}"
                        )

        train_indices = np.flatnonzero(
            assignments == train_value
        ).tolist()
        val_indices = np.flatnonzero(
            assignments == validation_value
        ).tolist()
        if not train_indices or not val_indices:
            raise RuntimeError(
                "预分配 split 为空："
                f"train={len(train_indices)}，validation={len(val_indices)}"
            )

        train_set = torch.utils.data.Subset(
            dataset,
            train_indices,
        )
        val_set = torch.utils.data.Subset(
            dataset,
            val_indices,
        )

        print("\n[preassigned split audit]")
        print("assignment_key:", assignment_key)
        print("train_value:", train_value, "episodes:", len(train_indices))
        print(
            "validation_value:",
            validation_value,
            "episodes:",
            len(val_indices),
        )
        reserved = int(
            len(assignments)
            - len(train_indices)
            - len(val_indices)
        )
        print("reserved episodes:", reserved)

    elif (
        stratify_key
        and dataset_file_path.is_file()
    ):
        # Primitive mixed 数据按 episode 的 motion_type 分层，
        # 避免 translation/rotation 在 train/validation 中比例漂移。
        with h5py.File(dataset_file_path, "r") as file:
            if stratify_key not in file:
                raise KeyError(
                    f"split.stratify_key={stratify_key!r}，"
                    "但 H5 中不存在该字段。"
                )

            episode_offsets = np.asarray(
                file["ep_offset"][:],
                dtype=np.int64,
            )

            episode_labels = np.asarray(
                file[stratify_key][episode_offsets]
            ).reshape(-1)

        rng = np.random.default_rng(seed)

        train_indices = []
        val_indices = []

        for label in np.unique(episode_labels):
            indices = np.flatnonzero(
                episode_labels == label
            )

            rng.shuffle(indices)

            n_train = int(
                round(
                    float(cfg.train_split)
                    * len(indices)
                )
            )

            train_indices.extend(
                indices[:n_train].tolist()
            )
            val_indices.extend(
                indices[n_train:].tolist()
            )

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        train_set = torch.utils.data.Subset(
            dataset,
            train_indices,
        )

        val_set = torch.utils.data.Subset(
            dataset,
            val_indices,
        )

        print("\n[stratified split audit]")

        for label in np.unique(episode_labels):
            train_count = int(
                np.sum(
                    episode_labels[
                        np.asarray(train_indices)
                    ] == label
                )
            )

            val_count = int(
                np.sum(
                    episode_labels[
                        np.asarray(val_indices)
                    ] == label
                )
            )

            print(
                f"label={int(label)} "
                f"train={train_count} "
                f"validation={val_count}"
            )

    else:
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[
                cfg.train_split,
                1.0 - cfg.train_split,
            ],
            generator=split_generator,
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=train_loader_generator,
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

    # --------------------------------------------------------------
    # Fail-fast architecture audit
    # --------------------------------------------------------------
    batchnorm_modules = [
        name
        for name, module in world_model.named_modules()
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        )
    ]

    layernorm_modules = [
        name
        for name, module in world_model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
    ]

    print("\n[architecture audit]")
    print("model class:", type(world_model).__name__)
    print("batchnorm_modules:", batchnorm_modules)
    print("layernorm_module_count:", len(layernorm_modules))
    strict_recurrent_bottleneck = bool(
        getattr(world_model, "strict_recurrent_bottleneck", False)
    )
    print("strict_recurrent_bottleneck:", strict_recurrent_bottleneck)
    print(
        "memory.initial_state_std:",
        float(world_model.memory.initial_state_std),
    )

    require_no_batchnorm = bool(
        cfg_select(
            cfg,
            "safety.require_no_batchnorm",
            True,
        )
    )

    if require_no_batchnorm and batchnorm_modules:
        raise RuntimeError(
            "ARCHITECTURE AUDIT FAILED: "
            "模型中仍存在 BatchNorm："
            f"{batchnorm_modules}"
        )

    require_strict_bottleneck = bool(
        cfg_select(
            cfg,
            "safety.require_strict_recurrent_bottleneck",
            False,
        )
    )
    if require_strict_bottleneck and not strict_recurrent_bottleneck:
        raise RuntimeError(
            "ARCHITECTURE AUDIT FAILED: strict recurrent bottleneck is required."
        )
    if require_strict_bottleneck:
        forbidden_predictor_modules = [
            f"{name}:{type(module).__name__}"
            for name, module in world_model.predictor.named_modules()
            if any(
                token in type(module).__name__.lower()
                for token in ("attention", "transformer", "arpredictor")
            )
        ]
        positional_parameters = [
            name
            for name, _ in world_model.predictor.named_parameters()
            if "pos" in name.lower()
        ]
        if forbidden_predictor_modules or positional_parameters:
            raise RuntimeError(
                "ARCHITECTURE AUDIT FAILED: predictor contains temporal machinery: "
                f"modules={forbidden_predictor_modules}, "
                f"positional_parameters={positional_parameters}"
            )
        print("predictor_temporal_modules: []")
        print("predictor_positional_parameters: []")

    expected_h0_std = float(
        cfg_select(
            cfg,
            "safety.expected_initial_state_std",
            0.01,
        )
    )

    actual_h0_std = float(
        world_model.memory.initial_state_std
    )

    if abs(actual_h0_std - expected_h0_std) > 1e-12:
        raise RuntimeError(
            "ARCHITECTURE AUDIT FAILED: "
            f"initial_state_std={actual_h0_std}，"
            f"期望 {expected_h0_std}。"
        )

    normalization_heads = ["projector"]
    if strict_recurrent_bottleneck:
        normalization_heads.append("predictor")
    else:
        normalization_heads.append("pred_proj")
    for head_name in normalization_heads:
        head = getattr(world_model, head_name)

        head_batchnorm = [
            module
            for module in head.modules()
            if isinstance(
                module,
                torch.nn.modules.batchnorm._BatchNorm,
            )
        ]

        head_layernorm = [
            module
            for module in head.modules()
            if isinstance(module, torch.nn.LayerNorm)
        ]

        if head_batchnorm or not head_layernorm:
            raise RuntimeError(
                "ARCHITECTURE AUDIT FAILED: "
                f"{head_name} 未正确使用 LayerNorm。"
            )

        print(
            f"{head_name} normalization: LayerNorm"
        )

    configured_loss_mode = str(
        cfg_select(
            cfg,
            "oot_loss.regularizer_mode",
            "legacy_global",
        )
    )
    configured_v2 = configured_loss_mode == "conditioned_vicreg_v2"

    print("\n[loss v2 audit]")
    print("regularizer_mode:", configured_loss_mode)
    print(
        "stop_gradient_target:",
        bool(
            cfg_select(
                cfg,
                "oot_loss.stop_gradient_target",
                configured_v2,
            )
        ),
    )
    print(
        "global_sigreg_weight:",
        float(
            cfg_select(
                cfg,
                "oot_loss.global_sigreg_weight",
                0.0 if configured_v2 else cfg.loss.sigreg.weight,
            )
        ),
    )
    print(
        "legacy_global_variance_weight:",
        float(
            cfg_select(
                cfg,
                "oot_loss.legacy_global_variance_weight",
                0.0
                if configured_v2
                else cfg_select(cfg, "oot_loss.variance_weight", 1.0),
            )
        ),
    )
    print(
        "conditioned_variance_weight:",
        float(
            cfg_select(
                cfg,
                "oot_loss.conditioned_variance_weight",
                25.0,
            )
        ),
    )
    print(
        "conditioned_covariance_weight:",
        float(
            cfg_select(
                cfg,
                "oot_loss.conditioned_covariance_weight",
                1.0,
            )
        ),
    )
    print(
        "content_contrastive_weight:",
        float(
            cfg_select(
                cfg,
                "oot_loss.content_contrastive_weight",
                1.0,
            )
        ),
    )
    print("content_contrastive_scope: same_timestep_only")

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
        FixedValidationRNGCallback(
            seed=seed + 100_000,
        ),
        FirstBackwardGradientAudit(),
    ]


    if bool(
        cfg_select(
            cfg,
            "safety.collapse_guard.enabled",
            False,
        )
    ):
        callbacks.append(
            RepresentationCollapseGuard(
                mse_threshold=float(
                    cfg_select(
                        cfg,
                        "safety.collapse_guard.mse_threshold",
                        1e-4,
                    )
                ),
                cosine_threshold=float(
                    cfg_select(
                        cfg,
                        "safety.collapse_guard.cosine_threshold",
                        0.8,
                    )
                ),
                patience=int(
                    cfg_select(
                        cfg,
                        "safety.collapse_guard.patience",
                        1,
                    )
                ),
            )
        )

    if bool(
        cfg_select(
            cfg,
            "safety.lr_scan_stop.enabled",
            False,
        )
    ):
        callbacks.append(
            StopAfterValidationEpochs(
                validation_epochs=int(
                    cfg_select(
                        cfg,
                        "safety.lr_scan_stop."
                        "validation_epochs",
                        1,
                    )
                )
            )
        )

    if bool(
        cfg_select(
            cfg,
            "checkpoint.export_weights",
            True,
        )
    ):
        callbacks.append(
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=int(
                    cfg_select(
                        cfg,
                        "checkpoint.export_every_n_epochs",
                        1,
                    )
                ),
            )
        )

    # 保存 validation overall latent MSE 最低的若干 checkpoint。
    #
    # checkpoint 选择只使用 validation set；
    # test set 仍只用于最终报告，避免 test-selection bias。
    if bool(cfg_select(cfg, "checkpoint.enabled", True)):
        checkpoint_dir_value = cfg_select(
            cfg,
            "checkpoint.dirpath",
            None,
        )

        if checkpoint_dir_value:
            checkpoint_dir = Path(str(checkpoint_dir_value))
        else:
            checkpoint_dir = run_dir / "best_checkpoints"

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=(
                    f"{cfg.output_model_name}-"
                    "best-{epoch:03d}-{step:06d}"
                ),
                monitor=str(
                    cfg_select(
                        cfg,
                        "checkpoint.monitor",
                        "validate/pred_loss_reappearance",
                    )
                ),
                mode=str(
                    cfg_select(
                        cfg,
                        "checkpoint.mode",
                        "min",
                    )
                ),
                save_top_k=int(
                    cfg_select(
                        cfg,
                        "checkpoint.save_top_k",
                        3,
                    )
                ),
                save_last=bool(
                    cfg_select(
                        cfg,
                        "checkpoint.save_last",
                        True,
                    )
                ),
                every_n_epochs=int(
                    cfg_select(
                        cfg,
                        "checkpoint.every_n_epochs",
                        1,
                    )
                ),
                auto_insert_metric_name=False,
            )
        )

    # 可通过命令行临时开启：
    #   ++analysis.enabled=true
    #   ++analysis.save_dir=outputs/oot_latents
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

    trainer_kwargs = OmegaConf.to_container(
        cfg.trainer,
        resolve=True,
    )

    if not isinstance(trainer_kwargs, dict):
        raise TypeError(
            "cfg.trainer 必须能解析为 dict。"
        )

    trainer_kwargs.setdefault(
        "num_sanity_val_steps",
        1,
    )

    # 防止 trainer 配置与显式参数重复传入同名字段。
    trainer_kwargs.pop(
        "enable_checkpointing",
        None,
    )

    enable_checkpointing = bool(
        cfg_select(
            cfg,
            "checkpoint.enabled",
            True,
        )
    )

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"

    # stable_pretraining 的 requeue checkpoint 是全局设置，
    # 不是 Manager.__init__ 参数。
    #
    # smoke:
    #   checkpoint.enabled=false -> 不保存 last.ckpt
    #
    # formal:
    #   checkpoint.enabled=true  -> 保存 last.ckpt，允许断点续训
    spt.set(
        requeue_checkpoint=enable_checkpointing,
    )

    print("\n[spt checkpoint audit]")
    print(
        "requeue_checkpoint:",
        enable_checkpointing,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=lightning_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()


if __name__ == "__main__":
    run()
