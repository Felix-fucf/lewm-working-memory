#!/usr/bin/env bash
# ==============================================================================
# check_oot_workspace.sh
#
# 用途：
#   在真正改动 LeWorldModel 工程前，检查：
#   1. 仓库目录是否正确；
#   2. 原始 train.py / module.py / jepa.py / 配置文件是否存在；
#   3. Python、PyTorch、CUDA 和关键依赖是否可用；
#   4. 数据目录下有哪些 .h5 / .lance 数据；
#   5. OOT 数据文件内部有哪些 key、shape 和 dtype；
#   6. 当前 Git 工作区是否已有未提交修改。
#
# 默认路径采用你此前的 AutoDL 目录结构；也可以用环境变量覆盖：
#   export LEWM_REPO=/你的/le-wm/路径
#   export LOCAL_DATASET_DIR=/你的/数据目录
#   export OOT_DATASET_FILE=/完整路径/oot_train.h5
#
# 推荐运行：
#   mkdir -p /root/autodl-tmp/world-model/logs
#   bash check_oot_workspace.sh 2>&1 | tee \
#     /root/autodl-tmp/world-model/logs/oot_workspace_check.txt
# ==============================================================================

set -u

REPO="${LEWM_REPO:-/root/autodl-tmp/world-model/repos/le-wm}"
DATA_ROOT="${LOCAL_DATASET_DIR:-/root/autodl-tmp/world-model/datasets}"
OOT_DATASET_FILE="${OOT_DATASET_FILE:-}"

line() {
    printf '%*s\n' 88 '' | tr ' ' '='
}

section() {
    echo
    line
    echo "$1"
    line
}

section "0. 基本路径"

echo "LEWM_REPO          = ${REPO}"
echo "LOCAL_DATASET_DIR  = ${DATA_ROOT}"
echo "OOT_DATASET_FILE   = ${OOT_DATASET_FILE:-<未指定，将自动搜索 .h5>}"
echo "当前 shell 路径     = $(pwd)"
echo "当前用户             = $(whoami)"
echo "当前时间             = $(date '+%Y-%m-%d %H:%M:%S %Z')"

if [[ ! -d "${REPO}" ]]; then
    echo
    echo "[ERROR] 找不到仓库目录：${REPO}"
    echo "请先设置：export LEWM_REPO=/实际的/le-wm/路径"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo
    echo "[WARN] 找不到数据目录：${DATA_ROOT}"
    echo "后续 HDF5 检查会跳过。"
fi

cd "${REPO}"

section "1. 仓库顶层文件与 Git 状态"

echo "[仓库顶层]"
find . -maxdepth 1 -mindepth 1 \
    -not -name ".git" \
    -printf '%y  %p\n' | sort

echo
echo "[关键文件是否存在]"
required_files=(
    "train.py"
    "module.py"
    "jepa.py"
    "utils.py"
    "config/train/lewm.yaml"
    "config/train/model/lewm.yaml"
)

for file in "${required_files[@]}"; do
    if [[ -e "${file}" ]]; then
        printf '[ OK ] %s\n' "${file}"
    else
        printf '[MISS] %s\n' "${file}"
    fi
done

echo
echo "[当前可能已有的 OOT / schema 自定义文件]"
find . -maxdepth 4 -type f \
    \( -iname '*oot*' -o -iname '*schema*' -o -iname 'train_*.py' -o -iname 'module_*.py' \) \
    -not -path './.git/*' \
    -printf '%p\n' | sort | sed -n '1,240p'

echo
echo "[Git 分支和未提交修改]"
if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
    git branch --show-current 2>/dev/null || true
    git status --short || true
else
    echo "[WARN] 当前目录不是 Git 仓库，或系统中没有 git。"
fi

section "2. config/train 目录结构"

if [[ -d config/train ]]; then
    find config/train -maxdepth 3 -type f -printf '%p\n' | sort
else
    echo "[WARN] 找不到 config/train。"
fi

section "3. Python、依赖和 CUDA"

echo "[Python 可执行文件]"
which python || true
python --version || true

python - <<'PY'
import importlib
import platform
import sys

print("sys.executable =", sys.executable)
print("platform       =", platform.platform())

modules = [
    "torch",
    "lightning",
    "hydra",
    "omegaconf",
    "stable_worldmodel",
    "stable_pretraining",
    "h5py",
    "einops",
    "numpy",
]

print("\n[关键 Python 包]")
for name in modules:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "<无 __version__>")
        print(f"[ OK ] {name:20s} {version}")
    except Exception as exc:
        print(f"[FAIL] {name:20s} {type(exc).__name__}: {exc}")

print("\n[PyTorch / CUDA]")
try:
    import torch

    print("torch version          =", torch.__version__)
    print("torch.cuda.is_available=", torch.cuda.is_available())
    print("CUDA runtime           =", torch.version.cuda)
    print("GPU count              =", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        memory_gb = prop.total_memory / 1024**3
        print(
            f"GPU {i}: {prop.name}; "
            f"compute capability={prop.major}.{prop.minor}; "
            f"memory={memory_gb:.2f} GiB"
        )
except Exception as exc:
    print("无法读取 PyTorch/CUDA 信息：", repr(exc))
PY

section "4. 原代码和自定义代码的语法检查"

files_to_compile=(train.py module.py jepa.py utils.py)

for extra in train_oot.py module_oot.py train_schema_rulecue_v2.py; do
    if [[ -f "${extra}" ]]; then
        files_to_compile+=("${extra}")
    fi
done

for file in "${files_to_compile[@]}"; do
    if [[ -f "${file}" ]]; then
        if python -m py_compile "${file}"; then
            echo "[ OK ] py_compile ${file}"
        else
            echo "[FAIL] py_compile ${file}"
        fi
    fi
done

section "5. 数据目录概览"

if [[ -d "${DATA_ROOT}" ]]; then
    echo "[数据文件大小]"
    find "${DATA_ROOT}" -maxdepth 3 \
        \( -type f -o -type d \) \
        \( -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.lance' -o -iname '*oot*' \) \
        -printf '%y  %s bytes  %p\n' | sort | sed -n '1,240p'

    echo
    echo "[数据目录总占用]"
    du -sh "${DATA_ROOT}" 2>/dev/null || true
else
    echo "[SKIP] 数据目录不存在。"
fi

section "6. HDF5 数据结构检查"

export CHECK_DATA_ROOT="${DATA_ROOT}"
export CHECK_OOT_DATASET_FILE="${OOT_DATASET_FILE}"

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

data_root = Path(os.environ["CHECK_DATA_ROOT"])
explicit_file = os.environ.get("CHECK_OOT_DATASET_FILE", "").strip()

try:
    import h5py
    import numpy as np
except Exception as exc:
    print("[SKIP] 无法导入 h5py/numpy：", repr(exc))
    raise SystemExit(0)

if explicit_file:
    candidates = [Path(explicit_file)]
else:
    if not data_root.exists():
        print("[SKIP] 数据目录不存在。")
        raise SystemExit(0)

    # 优先检查名字中带 oot 的文件；若没有，再检查普通 h5 文件。
    all_h5 = sorted(data_root.rglob("*.h5")) + sorted(data_root.rglob("*.hdf5"))
    oot_h5 = [p for p in all_h5 if "oot" in p.name.lower() or "occlu" in p.name.lower()]
    candidates = oot_h5 if oot_h5 else all_h5

if not candidates:
    print("没有找到 .h5 / .hdf5 文件。")
    raise SystemExit(0)

print(f"共找到 {len(candidates)} 个候选 HDF5 文件。")
print("为避免输出过长，本次最多详细检查前 8 个。\n")

# 这些 key 是后续 train_oot.py 最关心的。
required_or_useful = {
    "pixels",
    "action",
    "episode_id",
    "timestep",
    "phase_id",
    "object_visible",
    "is_visible",
    "visible",
    "is_reappearance",
    "valid_mask",
    "object_x",
    "object_y",
    "object_position",
}

def preview_dataset(ds, sample_n: int = 512) -> str:
    """只抽取少量元素检查 NaN 和数值范围，避免一次读入整个大数据集。"""
    if ds.size == 0:
        return "empty"

    if not np.issubdtype(ds.dtype, np.number):
        return "non-numeric"

    try:
        # 沿第一维抽样；标量数据集单独处理。
        if ds.ndim == 0:
            arr = np.asarray(ds[()])
        else:
            n = min(ds.shape[0], sample_n)
            arr = np.asarray(ds[:n])

        if arr.size == 0:
            return "empty sample"

        finite = np.isfinite(arr)
        nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0

        if finite.any():
            min_v = float(arr[finite].min())
            max_v = float(arr[finite].max())
            return f"sample_min={min_v:.5g}, sample_max={max_v:.5g}, sample_nan={nan_count}"
        return f"sample has no finite value, sample_nan={nan_count}"
    except Exception as exc:
        return f"preview failed: {type(exc).__name__}: {exc}"

for path in candidates[:8]:
    print("=" * 88)
    print("FILE:", path)
    if not path.exists():
        print("[MISS] 文件不存在")
        continue

    print("SIZE:", f"{path.stat().st_size / 1024**3:.3f} GiB")

    try:
        with h5py.File(path, "r") as f:
            print("ROOT ATTRS:", dict(f.attrs))

            found_leaf_keys: list[str] = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    found_leaf_keys.append(name)
                    print(
                        f"DATASET {name:42s} "
                        f"shape={str(obj.shape):24s} "
                        f"dtype={str(obj.dtype):12s} "
                        f"{preview_dataset(obj)}"
                    )
                elif isinstance(obj, h5py.Group):
                    if name:
                        print(f"GROUP   {name}")

            f.visititems(visitor)

            leaf_names = {name.split("/")[-1] for name in found_leaf_keys}
            present = sorted(required_or_useful & leaf_names)
            missing_core = sorted({"pixels", "action"} - leaf_names)

            print("\n[OOT 相关 key]")
            print("present =", present)
            print("missing core =", missing_core or "<none>")

            # 对 episode_id 做轻量检查；用于判断随机切片会不会跨 episode。
            episode_key = next(
                (name for name in found_leaf_keys if name.split("/")[-1] == "episode_id"),
                None,
            )
            if episode_key is not None:
                ds = f[episode_key]
                n = min(ds.shape[0], 200_000)
                episode_ids = np.asarray(ds[:n]).reshape(-1)
                print(
                    f"episode_id 前 {n} 项："
                    f"unique={len(np.unique(episode_ids))}, "
                    f"first={episode_ids[:10].tolist()}, "
                    f"last={episode_ids[-10:].tolist()}"
                )

    except Exception as exc:
        print("[FAIL] 无法读取：", type(exc).__name__, exc)

print("\n检查结束。请保留完整输出，后续据此确定 OOT 数据 YAML 的 key 与 num_steps。")
PY

section "7. 检查结论提示"

cat <<'TXT'
请重点确认输出中的以下信息：

1. train.py / module.py / jepa.py 是否全部存在且可 py_compile；
2. 当前仓库是否已经有未提交修改；
3. OOT 数据文件的真实文件名；
4. OOT 数据是否至少包含 pixels 和 action；
5. 是否包含 phase_id、is_reappearance、object_visible 等分析标签；
6. pixels 的 shape 是否带时间维，action 的最后一维是多少；
7. 每条训练样本是否会被稳定地限制在同一个 episode 内。

在这些信息确认前，不要删除或覆盖原始 train.py、module.py、jepa.py。
TXT
