"""
Compare label distributions between two LIBERO-format datasets.

Label definition follows the provided loader logic:
    label = [ee_pos(3), quat_from_axis_angle(4), gripper_state_first_finger/0.04]

Outputs:
    - t-SNE scatter plot (both datasets together)
    - Per-dimension histogram overlays
    - Numeric summary stats (mean/std/min/max)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

TSNE = None
try:
    sklearn_manifold = importlib.import_module("sklearn.manifold")
    TSNE = sklearn_manifold.TSNE
    SKLEARN_TSNE_AVAILABLE = True
except Exception:
    SKLEARN_TSNE_AVAILABLE = False


# ---------------------------
# Edit paths if needed.
DATASET_A_DIR = Path(
    "/media/czhang883/PORTABLE_SSD/libero_random_exploration/"
    "data_dir/scratch/libero/env_rand_samples/lb_randsam_goal_100ep_single_task"
)
DATASET_B_DIR = Path("/media/czhang883/PORTABLE_SSD/LIBERO/libero/datasets/libero_goal")
OUT_DIR = Path(
    "/media/czhang883/PORTABLE_SSD/libero_random_exploration/data_dir/scratch/libero/tsne_compare"
)
# ---------------------------

SEED = 27
MAX_POINTS_PER_DATASET = 20000
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000


@dataclass
class DatasetStats:
    name: str
    labels: np.ndarray
    total_frames: int
    num_files: int


def axis_angle_to_quat_batch(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (rotation vector, shape [T, 3]) to quaternion [x, y, z, w].
    """
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(f"Expected axis-angle shape [T,3], got {axis_angle.shape}")
    return R.from_rotvec(axis_angle).as_quat()


def find_hdf5_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*.hdf5"))


def load_labels_from_hdf5_file(h5_path: Path) -> np.ndarray:
    """
    Read all demos from a single LIBERO-format file and return labels [N, 8].
    """
    all_labels = []
    with h5py.File(h5_path.as_posix(), "r") as f:
        if "data" not in f:
            return np.empty((0, 8), dtype=np.float64)

        data = f["data"]
        demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
        for demo_key in demo_keys:
            obs = data[demo_key]["obs"]
            if "ee_pos" not in obs or "ee_ori" not in obs or "gripper_states" not in obs:
                continue

            ee_pos = np.array(obs["ee_pos"], dtype=np.float64)
            ee_ori = np.array(obs["ee_ori"], dtype=np.float64)
            gripper_states = np.array(obs["gripper_states"], dtype=np.float64)
            if ee_pos.size == 0:
                continue

            t = min(len(ee_pos), len(ee_ori), len(gripper_states))
            ee_pos = ee_pos[:t]
            ee_ori = ee_ori[:t]
            gripper_states = gripper_states[:t]

            ee_quat = axis_angle_to_quat_batch(ee_ori)  # [T,4]
            q2s = gripper_states[:, 0:1] / 0.04
            labels = np.concatenate([ee_pos, ee_quat, q2s], axis=1)
            all_labels.append(labels)

    if not all_labels:
        return np.empty((0, 8), dtype=np.float64)
    return np.concatenate(all_labels, axis=0)


def load_dataset_labels(dataset_dir: Path, name: str) -> DatasetStats:
    files = find_hdf5_files(dataset_dir)
    print(f"[{name}] found {len(files)} hdf5 files in {dataset_dir}")
    labels_per_file = []
    for h5_path in tqdm(files, desc=f"Loading {name}", unit="file"):
        labels = load_labels_from_hdf5_file(h5_path)
        if labels.size > 0:
            labels_per_file.append(labels)

    if not labels_per_file:
        raise RuntimeError(f"[{name}] no valid labels loaded from {dataset_dir}")

    labels_all = np.concatenate(labels_per_file, axis=0)
    return DatasetStats(
        name=name,
        labels=labels_all,
        total_frames=labels_all.shape[0],
        num_files=len(files),
    )


def maybe_subsample(labels: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if labels.shape[0] <= max_points:
        return labels
    idx = rng.choice(labels.shape[0], size=max_points, replace=False)
    return labels[idx]


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std


def pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comp = vt[:2].T
    return x0 @ comp


def save_summary(stats_a: DatasetStats, stats_b: DatasetStats, out_dir: Path) -> None:
    out_path = out_dir / "label_summary.txt"
    names = ["ee_x", "ee_y", "ee_z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper_norm"]

    with open(out_path, "w") as f:
        for stats in [stats_a, stats_b]:
            f.write(f"=== {stats.name} ===\n")
            f.write(f"num_files: {stats.num_files}\n")
            f.write(f"num_frames: {stats.total_frames}\n")
            mean = stats.labels.mean(axis=0)
            std = stats.labels.std(axis=0)
            vmin = stats.labels.min(axis=0)
            vmax = stats.labels.max(axis=0)
            for i, name in enumerate(names):
                f.write(
                    f"{name:>12s} | mean={mean[i]: .6f} std={std[i]: .6f} "
                    f"min={vmin[i]: .6f} max={vmax[i]: .6f}\n"
                )
            f.write("\n")
    print(f"[save] {out_path}")


def plot_histograms(stats_a: DatasetStats, stats_b: DatasetStats, out_dir: Path) -> None:
    names = ["ee_x", "ee_y", "ee_z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper_norm"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        ax.hist(stats_a.labels[:, i], bins=120, alpha=0.5, density=True, label=stats_a.name)
        ax.hist(stats_b.labels[:, i], bins=120, alpha=0.5, density=True, label=stats_b.name)
        ax.set_title(name)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend()

    fig.suptitle("Per-dimension Label Distributions")
    fig.tight_layout()
    out_path = out_dir / "label_hist_compare.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[save] {out_path}")


def run_tsne(stats_a: DatasetStats, stats_b: DatasetStats, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    a_sub = maybe_subsample(stats_a.labels, MAX_POINTS_PER_DATASET, rng)
    b_sub = maybe_subsample(stats_b.labels, MAX_POINTS_PER_DATASET, rng)

    x = np.concatenate([a_sub, b_sub], axis=0)
    y = np.array([0] * len(a_sub) + [1] * len(b_sub), dtype=np.int64)

    x_std = standardize(x)
    if SKLEARN_TSNE_AVAILABLE:
        tsne = TSNE(
            n_components=2,
            perplexity=TSNE_PERPLEXITY,
            random_state=SEED,
            init="pca",
            learning_rate="auto",
            n_iter=TSNE_N_ITER,
            verbose=1,
        )
        z = tsne.fit_transform(x_std)
        method_name = "t-SNE"
    else:
        # Fallback so script still runs if sklearn is unavailable.
        print("[warn] sklearn not found, falling back to PCA-2D projection.")
        z = pca_2d(x_std)
        method_name = "PCA (fallback, sklearn missing)"

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    m0 = y == 0
    m1 = y == 1
    ax.scatter(z[m0, 0], z[m0, 1], s=2, alpha=0.4, label=f"{stats_a.name} ({len(a_sub)})")
    ax.scatter(z[m1, 0], z[m1, 1], s=2, alpha=0.4, label=f"{stats_b.name} ({len(b_sub)})")
    ax.set_title(f"{method_name} of Labels (Combined)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend()
    ax.grid(alpha=0.2)

    out_path = out_dir / "label_tsne_compare.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    if not DATASET_A_DIR.exists():
        raise FileNotFoundError(DATASET_A_DIR)
    if not DATASET_B_DIR.exists():
        raise FileNotFoundError(DATASET_B_DIR)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_DIR / f"tsne_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_a = load_dataset_labels(DATASET_A_DIR, "custom_single_task")
    stats_b = load_dataset_labels(DATASET_B_DIR, "libero_goal_official")

    print(
        f"[frames] {stats_a.name}={stats_a.total_frames}, "
        f"{stats_b.name}={stats_b.total_frames}"
    )

    save_summary(stats_a, stats_b, out_dir)
    plot_histograms(stats_a, stats_b, out_dir)
    run_tsne(stats_a, stats_b, out_dir)
    print(f"[done] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
