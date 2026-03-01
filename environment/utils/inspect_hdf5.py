from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


# Single-purpose script by design (no CLI args).
INPUT_H5 = Path(
    "/media/czhang883/PORTABLE_SSD/libero_random_exploration/data_dir/scratch/libero/env_rand_samples/lb_randsam_spatial_3ep_20260301_175400.hdf5"
)
OUTPUT_MP4 = INPUT_H5.with_name(f"{INPUT_H5.stem}_all_rgb_concat.mp4")
FPS = 30
SPEEDUP = 4  # keep 1 to inspect every frame
SPEED_MODE = "stride"  # "stride" or "fps"


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype in (np.float32, np.float64):
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255)
        return frame.astype(np.uint8)
    return frame.astype(np.uint8)


def _is_rgb_dataset(ds: h5py.Dataset) -> bool:
    shape = ds.shape
    if len(shape) != 4:
        return False
    return shape[-1] == 3 or shape[1] == 3


def _iter_rgb_frames(ds: h5py.Dataset):
    shape = ds.shape
    channels_last = shape[-1] == 3
    channels_first = shape[1] == 3
    if not (channels_last or channels_first):
        raise ValueError(f"Unsupported RGB dataset shape: {shape}")

    for i in range(shape[0]):
        frame = ds[i]
        if channels_first:
            frame = frame.transpose(1, 2, 0)
        yield _to_uint8(frame)


def _find_rgb_dataset_paths(h5f: h5py.File) -> list[str]:
    paths: list[str] = []

    def _visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and _is_rgb_dataset(obj):
            paths.append(name)

    h5f.visititems(_visitor)
    return sorted(paths)


def export_all_rgb_to_one_video(
    h5_path: Path,
    out_path: Path,
    fps: int,
    speedup: int,
    speed_mode: str,
) -> Path:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    out_fps = fps * speedup if speed_mode == "fps" else fps
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path.as_posix(), "r") as h5f:
        dataset_paths = _find_rgb_dataset_paths(h5f)
        if not dataset_paths:
            raise ValueError(f"No RGB datasets found in {h5_path}")

        print(f"[info] Found {len(dataset_paths)} RGB datasets")
        input_frames = 0
        written_frames = 0

        with imageio.get_writer(out_path.as_posix(), fps=out_fps, codec="libx264", macro_block_size=None) as writer:
            for dpath in dataset_paths:
                ds = h5f[dpath]
                print(f"[dataset] {dpath} shape={ds.shape}")
                for frame in _iter_rgb_frames(ds):
                    if speedup > 1 and speed_mode == "stride" and (input_frames % speedup != 0):
                        input_frames += 1
                        continue
                    writer.append_data(frame)
                    written_frames += 1
                    input_frames += 1

    duration = written_frames / float(max(out_fps, 1))
    print(
        f"[done] {out_path} | datasets={len(dataset_paths)} "
        f"input_frames={input_frames} written_frames={written_frames} fps={out_fps} duration={duration:.2f}s"
    )
    return out_path


def main() -> None:
    export_all_rgb_to_one_video(
        h5_path=INPUT_H5,
        out_path=OUTPUT_MP4,
        fps=FPS,
        speedup=SPEEDUP,
        speed_mode=SPEED_MODE,
    )


if __name__ == "__main__":
    main()