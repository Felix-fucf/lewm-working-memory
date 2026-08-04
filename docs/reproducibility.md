# 可复现性与 artifact 说明

## 代码来源

本分支基于上游 `lucas-maes/le-wm` 的 `e80f2df`，保留完整 Git 历史。正式实验代码来自服务器上的 phase-II 快照；公开前只做了机器路径泛化、文档状态更新和结果精选。原始与公开文件的 SHA256 对照见 `results/manifests/source-provenance.csv`。

## 已核验环境

- Python 3.12.3
- PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128
- Lightning 2.6.5
- stable-pretraining 0.1.7
- stable-worldmodel 0.1.1
- Hydra 1.3.4 / OmegaConf 2.3.1
- NumPy 2.5.1 / h5py 3.16.0

完整核心版本见 `requirements.txt`。CUDA wheel 需要从 PyTorch 的 cu128 index 安装；CPU 或其他 CUDA 版本可以替换 PyTorch wheel，但不应把环境差异误当作实验复现。

## 本地目录约定

```bash
export WORLD_MODEL_ROOT="$(pwd)"
export LOCAL_DATASET_DIR="$WORLD_MODEL_ROOT/datasets"
export STABLEWM_HOME="$WORLD_MODEL_ROOT/.stable-wm"
```

数据集、checkpoint 和完整训练日志不进入 GitHub。请根据 `results/manifests/` 中的 SHA256 与 run record，把私有归档中的文件放回上述目录。

## 测试

```bash
python -m pytest -q -p no:cacheprovider tests
```

整理前 phase-II 快照与整理后的公开分支均应为 `15 passed`。

## 重建 V3 selector

公开仓库保留 selector 所需的两套 validation raw audit：

```bash
python analysis/select_occlusion_v3_winner.py \
  --campaign-root results/manifests/v3-selector-input \
  --output /tmp/selection_decision.json
```

预期输出为 `selected: null`。仓库内的生成件位于 `results/manifests/selection_decision.json`，其来源说明位于相邻 provenance JSON。

## 训练入口

- C0–R3：`python scripts/run_occlusion_campaign.py --world-root "$WORLD_MODEL_ROOT" <stage>`
- V3：`python scripts/run_occlusion_v3_campaign.py --world-root "$WORLD_MODEL_ROOT" <stage>`

详细 gate 与运行顺序见 `configs/occlusion_curriculum/RUNBOOK.md` 和 `configs/occlusion_v3/RUNBOOK_V3.md`。本仓库的主参考配置是 V3 N1；N2/N3 是 ablation，不是优胜模型。

## Artifact 边界

- GitHub：源码、配置、测试、summary.csv、精选 raw audit、SVG 图和 SHA256 manifest。
- 远程归档：C0/R1/R2/R3/N1/N2/N3 最终 `.pt` 与 `last.ckpt`，main25 的 epoch 18/23/25，早期 E15/e100 最终 checkpoint，以及 primitive-v1/v2/v3 核心数据集。
- 不公开：HDF5、checkpoint、NPZ、完整日志、Lightning/Hydra outputs、压缩迁移包和任何服务器路径。
