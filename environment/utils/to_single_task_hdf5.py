"""
Convert random-sampled LIBERO goal HDF5 into per-task LIBERO-style demo files.

Input (current random sampler output):
    <task_name>/<episode_idx>/{action, agentview_image, ee_poses}

Output (per task, LIBERO-style keys):
    data/demo_i/
        actions
        dones
        rewards
        robot_states
        states
        obs/
            agentview_rgb
            eye_in_hand_rgb
            ee_pos
            ee_ori
            ee_states
            gripper_states
            joint_states

This script is intentionally no-args. Edit paths below if needed, then run:
    python environment/utils/to_single_task_hdf5.py
"""

from pathlib import Path

import h5py
import numpy as np


# ---------------------------
# Edit these paths if needed.
INPUT_H5 = Path(
    "/media/czhang883/PORTABLE_SSD/libero_random_exploration/"
    "data_dir/scratch/libero/env_rand_samples/lb_randsam_goal_100ep_20260301_042335.hdf5"
)
OUTPUT_DIR = Path(
    "/media/czhang883/PORTABLE_SSD/libero_random_exploration/"
    "data_dir/scratch/libero/env_rand_samples/lb_randsam_goal_100ep_single_task"
)
LIBERO_TEMPLATE_DIR = Path(
    "/media/czhang883/PORTABLE_SSD/LIBERO/libero/datasets/libero_goal"
)
# ---------------------------


def task_to_filename(task_name: str) -> str:
    return f"{task_name.replace(' ', '_')}_demo.hdf5"


def _normalize_lengths(
    imgs: np.ndarray,
    acts: np.ndarray,
    ee_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match LIBERO's convention: per-step arrays share same length T.
    Random samples often store obs with T+1 and actions with T.
    """
    t = min(len(acts), len(imgs), len(ee_pos))
    return imgs[:t], acts[:t], ee_pos[:t]


def _gripper_action_to_states(gripper_action: np.ndarray) -> np.ndarray:
    """
    Map action[-1] in [-1, 1] to a plausible 2-finger state in meters.
    Loader uses first channel and divides by 0.04.
    """
    g = np.clip(gripper_action.astype(np.float64), -1.0, 1.0)
    open_width = (g + 1.0) * 0.5 * 0.04  # [0, 0.04]
    return np.stack([open_width, -open_width], axis=1)


def _copy_template_data_attrs_if_exists(task_name: str, data_group: h5py.Group) -> None:
    """
    Copy LIBERO task-level attrs (env_args, bddl_file_name, etc.) when available.
    """
    template_path = LIBERO_TEMPLATE_DIR / task_to_filename(task_name)
    if not template_path.exists():
        return

    with h5py.File(template_path.as_posix(), "r") as tf:
        if "data" in tf:
            for k, v in tf["data"].attrs.items():
                data_group.attrs[k] = v


def convert() -> None:
    if not INPUT_H5.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_H5}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(INPUT_H5.as_posix(), "r") as src:
        task_names = sorted(src.keys())
        print(f"[convert] input={INPUT_H5}")
        print(f"[convert] found {len(task_names)} tasks")

        for task_name in task_names:
            task_group = src[task_name]
            ep_keys = sorted(task_group.keys(), key=lambda x: int(x))
            out_path = OUTPUT_DIR / task_to_filename(task_name)

            with h5py.File(out_path.as_posix(), "w") as dst:
                data = dst.create_group("data")
                _copy_template_data_attrs_if_exists(task_name, data)

                total_samples = 0
                for i_demo, ep_key in enumerate(ep_keys):
                    ep = task_group[ep_key]
                    imgs = np.array(ep["agentview_image"], dtype=np.uint8)
                    acts = np.array(ep["action"], dtype=np.float64)
                    ee_pos = np.array(ep["ee_poses"], dtype=np.float64)
                    imgs, acts, ee_pos = _normalize_lengths(imgs, acts, ee_pos)

                    t = len(acts)
                    if t == 0:
                        print(f"[skip] {task_name}/{ep_key} has 0 aligned frames")
                        continue

                    demo = data.create_group(f"demo_{i_demo}")
                    obs = demo.create_group("obs")

                    # Core keys your loader uses.
                    obs.create_dataset("agentview_rgb", data=imgs, compression="gzip")
                    obs.create_dataset("ee_pos", data=ee_pos)
                    obs.create_dataset("ee_ori", data=np.zeros((t, 3), dtype=np.float64))
                    obs.create_dataset("gripper_states", data=_gripper_action_to_states(acts[:, 6]))

                    # Extra LIBERO-like keys for compatibility with other tools.
                    obs.create_dataset("eye_in_hand_rgb", data=np.zeros_like(imgs), compression="gzip")
                    obs.create_dataset("ee_states", data=np.concatenate([ee_pos, np.zeros((t, 3))], axis=1))
                    obs.create_dataset("joint_states", data=np.zeros((t, 7), dtype=np.float64))

                    demo.create_dataset("actions", data=acts)
                    demo.create_dataset("dones", data=np.array([0] * (t - 1) + [1], dtype=np.uint8))
                    demo.create_dataset("rewards", data=np.zeros((t,), dtype=np.uint8))
                    demo.create_dataset("robot_states", data=np.zeros((t, 9), dtype=np.float64))
                    demo.create_dataset("states", data=np.zeros((t, 79), dtype=np.float64))

                    # Match key names seen in official files.
                    demo.attrs["num_samples"] = int(t)
                    if "env_seed" in ep.attrs:
                        demo.attrs["env_seed"] = ep.attrs["env_seed"]
                    if "env_list_name" in ep.attrs:
                        demo.attrs["env_list_name"] = ep.attrs["env_list_name"]

                    total_samples += t

                data.attrs["num_demos"] = int(len(ep_keys))
                data.attrs["total"] = int(total_samples)

            print(f"[write] {out_path} | demos={len(ep_keys)} total_samples={total_samples}")

    print(f"[done] output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert()
