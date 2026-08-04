from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(os.environ.get("LOCAL_DATASET_DIR", "datasets")) / "primitive_v1/formal_v1"

TRANSLATION_PATH = (
    ROOT / "primitive_v1_translation_train.h5"
)

ROTATION_PATH = (
    ROOT / "primitive_v1_rotation_train.h5"
)

OUTPUT_PATH = (
    ROOT / "primitive_v1_mixed_train.h5"
)

SEED = 20260713
EPISODE_LENGTH = 22


def dataset_frame_count(
    file: h5py.File,
    key: str,
) -> int:
    return int(file[key].shape[0])


with (
    h5py.File(TRANSLATION_PATH, "r") as ft,
    h5py.File(ROTATION_PATH, "r") as fr,
):
    n_translation = int(
        ft.attrs["num_episodes"]
    )
    n_rotation = int(
        fr.attrs["num_episodes"]
    )

    assert n_translation == 14000
    assert n_rotation == 10000

    assert int(
        ft.attrs["episode_length"]
    ) == EPISODE_LENGTH

    assert int(
        fr.attrs["episode_length"]
    ) == EPISODE_LENGTH

    translation_keys = set(ft.keys())
    rotation_keys = set(fr.keys())

    common_keys = sorted(
        translation_keys & rotation_keys
    )

    assert "pixels" in common_keys
    assert "action" in common_keys
    assert "frame_loss_weight" in common_keys
    assert "motion_type" in common_keys
    assert "episode_id" in common_keys

    # 只保留真正以 frame 为第一维的数据。
    common_frame_keys = []

    for key in common_keys:
        expected_t = (
            n_translation
            * EPISODE_LENGTH
        )

        expected_r = (
            n_rotation
            * EPISODE_LENGTH
        )

        if (
            dataset_frame_count(ft, key)
            == expected_t
            and dataset_frame_count(fr, key)
            == expected_r
        ):
            common_frame_keys.append(key)

    print("Common frame keys:")
    for key in common_frame_keys:
        print(" ", key)

    rng = np.random.default_rng(SEED)

    # modality:
    #   0 = translation
    #   1 = rotation
    modality = np.concatenate(
        [
            np.zeros(
                n_translation,
                dtype=np.int8,
            ),
            np.ones(
                n_rotation,
                dtype=np.int8,
            ),
        ]
    )

    source_episode = np.concatenate(
        [
            np.arange(
                n_translation,
                dtype=np.int64,
            ),
            np.arange(
                n_rotation,
                dtype=np.int64,
            ),
        ]
    )

    permutation = rng.permutation(
        n_translation + n_rotation
    )

    modality = modality[permutation]
    source_episode = source_episode[
        permutation
    ]

    n_total = (
        n_translation + n_rotation
    )

    total_frames = (
        n_total * EPISODE_LENGTH
    )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with h5py.File(
        OUTPUT_PATH,
        "w",
        libver="latest",
    ) as output:
        # ----------------------------------------------
        # Copy and update attributes
        # ----------------------------------------------
        for key, value in ft.attrs.items():
            output.attrs[key] = value

        output.attrs["num_episodes"] = n_total
        output.attrs[
            "completed_episodes"
        ] = n_total
        output.attrs["completed"] = 1
        output.attrs[
            "episode_length"
        ] = EPISODE_LENGTH

        output.attrs[
            "dataset_variant"
        ] = "primitive_v1_mixed_train"

        output.attrs[
            "mix_seed"
        ] = SEED

        output.attrs[
            "translation_episodes"
        ] = n_translation

        output.attrs[
            "rotation_episodes"
        ] = n_rotation

        output.attrs[
            "translation_fraction"
        ] = n_translation / n_total

        output.attrs[
            "rotation_fraction"
        ] = n_rotation / n_total

        output.attrs[
            "translation_source"
        ] = str(TRANSLATION_PATH)

        output.attrs[
            "rotation_source"
        ] = str(ROTATION_PATH)

        # 保存 mixed episode 到源 episode 的映射。
        output.create_dataset(
            "mixed_source_modality",
            data=modality,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        output.create_dataset(
            "mixed_source_episode",
            data=source_episode,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # ----------------------------------------------
        # Episode index required by stable_worldmodel
        # ----------------------------------------------
        #
        # ep_len[i] 是第 i 个 episode 的长度；
        # ep_offset[i] 是它在逐帧数组中的起始位置。
        ep_len_dtype = ft["ep_len"].dtype
        ep_offset_dtype = ft["ep_offset"].dtype

        assert ep_len_dtype == fr["ep_len"].dtype
        assert ep_offset_dtype == fr["ep_offset"].dtype

        output.create_dataset(
            "ep_len",
            data=np.full(
                n_total,
                EPISODE_LENGTH,
                dtype=ep_len_dtype,
            ),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        output.create_dataset(
            "ep_offset",
            data=(
                np.arange(
                    n_total,
                    dtype=ep_offset_dtype,
                )
                * EPISODE_LENGTH
            ),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # ----------------------------------------------
        # pixels: Virtual Dataset
        # ----------------------------------------------
        pixel_shape_t = ft["pixels"].shape
        pixel_shape_r = fr["pixels"].shape

        assert (
            pixel_shape_t[1:]
            == pixel_shape_r[1:]
        )

        pixel_dtype = ft["pixels"].dtype

        assert pixel_dtype == fr["pixels"].dtype

        layout = h5py.VirtualLayout(
            shape=(
                total_frames,
                *pixel_shape_t[1:],
            ),
            dtype=pixel_dtype,
        )

        source_t = h5py.VirtualSource(
            str(TRANSLATION_PATH),
            "pixels",
            shape=pixel_shape_t,
        )

        source_r = h5py.VirtualSource(
            str(ROTATION_PATH),
            "pixels",
            shape=pixel_shape_r,
        )

        for dst_episode in range(n_total):
            src_episode = int(
                source_episode[dst_episode]
            )

            dst_start = (
                dst_episode * EPISODE_LENGTH
            )
            dst_end = (
                dst_start + EPISODE_LENGTH
            )

            src_start = (
                src_episode * EPISODE_LENGTH
            )
            src_end = (
                src_start + EPISODE_LENGTH
            )

            if modality[dst_episode] == 0:
                layout[
                    dst_start:dst_end
                ] = source_t[
                    src_start:src_end
                ]
            else:
                layout[
                    dst_start:dst_end
                ] = source_r[
                    src_start:src_end
                ]

        output.create_virtual_dataset(
            "pixels",
            layout,
            fillvalue=0,
        )

        # ----------------------------------------------
        # Other common metadata: physical small arrays
        # ----------------------------------------------
        for key in common_frame_keys:
            if key == "pixels":
                continue

            ds_t = ft[key]
            ds_r = fr[key]

            if (
                ds_t.dtype != ds_r.dtype
                or ds_t.shape[1:]
                != ds_r.shape[1:]
            ):
                raise RuntimeError(
                    f"Incompatible field {key}: "
                    f"translation={ds_t.shape}/{ds_t.dtype}, "
                    f"rotation={ds_r.shape}/{ds_r.dtype}"
                )

            tail_shape = ds_t.shape[1:]

            arr_t = np.asarray(
                ds_t
            ).reshape(
                n_translation,
                EPISODE_LENGTH,
                *tail_shape,
            )

            arr_r = np.asarray(
                ds_r
            ).reshape(
                n_rotation,
                EPISODE_LENGTH,
                *tail_shape,
            )

            destination = np.empty(
                (
                    n_total,
                    EPISODE_LENGTH,
                    *tail_shape,
                ),
                dtype=ds_t.dtype,
            )

            t_positions = np.flatnonzero(
                modality == 0
            )

            r_positions = np.flatnonzero(
                modality == 1
            )

            destination[t_positions] = (
                arr_t[
                    source_episode[t_positions]
                ]
            )

            destination[r_positions] = (
                arr_r[
                    source_episode[r_positions]
                ]
            )

            # 混合后重新赋予唯一 episode_id。
            if key == "episode_id":
                episode_ids = np.arange(
                    n_total,
                    dtype=ds_t.dtype,
                )

                destination = np.repeat(
                    episode_ids[:, None],
                    EPISODE_LENGTH,
                    axis=1,
                )

                if tail_shape:
                    destination = (
                        destination.reshape(
                            n_total,
                            EPISODE_LENGTH,
                            *tail_shape,
                        )
                    )

            flat = destination.reshape(
                total_frames,
                *tail_shape,
            )

            chunk_frames = min(
                4096,
                total_frames,
            )

            chunks = (
                chunk_frames,
                *tail_shape,
            )

            output.create_dataset(
                key,
                data=flat,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            print(
                f"written {key}: "
                f"{flat.shape} {flat.dtype}"
            )


print("\nCreated:")
print(OUTPUT_PATH)
print(
    "file size:",
    OUTPUT_PATH.stat().st_size
    / 1024**2,
    "MiB",
)


# ============================================================
# Validation
# ============================================================
with h5py.File(OUTPUT_PATH, "r") as mixed:
    assert mixed["pixels"].is_virtual

    assert int(
        mixed.attrs["num_episodes"]
    ) == 24000

    assert mixed["pixels"].shape[0] == (
        24000 * EPISODE_LENGTH
    )

    assert int(
        np.sum(
            mixed[
                "mixed_source_modality"
            ][:] == 0
        )
    ) == 14000

    assert int(
        np.sum(
            mixed[
                "mixed_source_modality"
            ][:] == 1
        )
    ) == 10000

    weights = np.asarray(
        mixed["frame_loss_weight"][:22]
    )

    assert abs(
        float(weights.sum()) - 20.2
    ) < 1e-4

    assert abs(
        float(weights[1:].sum()) - 19.2
    ) < 1e-4

    episode_ids = np.asarray(
        mixed["episode_id"]
    ).reshape(24000, 22)

    assert np.array_equal(
        episode_ids[:, 0],
        np.arange(24000),
    )

    assert np.all(
        episode_ids
        == episode_ids[:, :1]
    )

    # 读取若干虚拟图像，确认 VDS 链接可用。
    probe_episodes = [
        0,
        1,
        100,
        1000,
        23999,
    ]

    for episode in probe_episodes:
        frame = np.asarray(
            mixed["pixels"][
                episode * 22
            ]
        )

        assert frame.size > 0
        assert np.isfinite(frame).all()

    print("\nPASS: mixed VDS validation")
