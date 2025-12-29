#!/usr/bin/env python3
"""Merge multiple PickPlaceBottle datasets into a single LeRobot dataset.

This variant expects every input directory to already be a fully materialized
LeRobot dataset (with ``data/``, ``videos/``, and ``meta`` folders). It copies
all parquet and video files, rebuilds the metadata, and recomputes
normalization statistics so the merged dataset can be used directly for
training.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_INPUTS = [
    Path("/mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v2"),
    Path("/mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v3"),
]
DEFAULT_OUTPUT = Path("/mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_MegaMerge")
METADATA_DIRNAME = "meta"
DATA_SUBDIR = "data"
VIDEOS_SUBDIR = "videos"
STATS_REL_PATH = Path(METADATA_DIRNAME) / "stats.json"


@dataclass
class EpisodeRecord:
    episode_index: int
    tasks: list[str]
    length: int


class StatsAccumulator:
    """Streaming tracker for mean/std and percentile estimates."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.quantile_chunks: list[np.ndarray] = []

    def update(self, batch: np.ndarray) -> None:
        array = np.asarray(batch, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.size == 0:
            return
        if self.mean is None:
            feature_dim = array.shape[1]
            self.mean = np.zeros(feature_dim, dtype=np.float64)
            self.m2 = np.zeros(feature_dim, dtype=np.float64)
        batch_count = array.shape[0]
        batch_mean = array.mean(axis=0)
        centered = array - batch_mean
        batch_m2 = (centered * centered).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + (delta * delta) * self.count * batch_count / total
        self.count = total
        self.quantile_chunks.append(array.astype(np.float32, copy=False))

    def finalize(self) -> dict[str, list[float]]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("StatsAccumulator received no data")
        std = np.sqrt(self.m2 / self.count)
        values = np.concatenate(self.quantile_chunks, axis=0)
        q01 = np.quantile(values, 0.01, axis=0)
        q99 = np.quantile(values, 0.99, axis=0)
        self.quantile_chunks.clear()
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge full PickPlaceBottle datasets")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Absolute paths to the dataset roots that should be merged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination directory for the merged dataset.",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        type=int,
        default=None,
        help="Subset of camera IDs to retain. Defaults to all RGB cameras present in the first dataset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override the chunk size. By default, the value from the first dataset is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory if it already exists.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy preview JPG/PNG images if they exist.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(root: Path) -> tuple[dict, list[dict], list[dict]]:
    meta_dir = root / METADATA_DIRNAME
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    episodes = read_jsonl(meta_dir / "episodes.jsonl")
    return info, tasks, episodes


def detect_camera_ids(info: dict) -> list[int]:
    ids: set[int] = set()
    for key in info.get("features", {}):
        if key.startswith("observation.camera_") and key.endswith(".rgb"):
            try:
                cam_id = int(key.split("_")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            ids.add(cam_id)
    if not ids:
        raise ValueError("No RGB camera features detected in metadata")
    return sorted(ids)


def copy_episode(
    source_root: Path,
    target_root: Path,
    local_idx: int,
    global_idx: int,
    chunk_size: int,
    camera_ids: Sequence[int],
    copy_images: bool,
) -> None:
    src_chunk = local_idx // chunk_size
    dst_chunk = global_idx // chunk_size
    src_data = source_root / DATA_SUBDIR / f"chunk-{src_chunk:03d}" / f"episode_{local_idx:06d}.parquet"
    dst_data_dir = target_root / DATA_SUBDIR / f"chunk-{dst_chunk:03d}"
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_data, dst_data_dir / f"episode_{global_idx:06d}.parquet")

    for cam_id in camera_ids:
        src_video = (
            source_root
            / VIDEOS_SUBDIR
            / f"chunk-{src_chunk:03d}"
            / f"observation.camera_{cam_id}.rgb"
            / f"episode_{local_idx:06d}.mp4"
        )
        dst_video_dir = (
            target_root
            / VIDEOS_SUBDIR
            / f"chunk-{dst_chunk:03d}"
            / f"observation.camera_{cam_id}.rgb"
        )
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video_dir / f"episode_{global_idx:06d}.mp4")

        if copy_images:
            src_img = (
                source_root
                / "images"
                / f"observation.camera_{cam_id}.rgb"
                / f"episode_{local_idx:06d}.jpg"
            )
            if not src_img.exists():
                src_img = src_img.with_suffix(".png")
            if src_img.exists():
                dst_img_dir = target_root / "images" / f"observation.camera_{cam_id}.rgb"
                dst_img_dir.mkdir(parents=True, exist_ok=True)
                dst_img = dst_img_dir / src_img.name.replace(f"{local_idx:06d}", f"{global_idx:06d}")
                shutil.copy2(src_img, dst_img)


def compute_dataset_stats(data_root: Path) -> dict[str, dict[str, list[float]]]:
    accumulators = {
        "state": StatsAccumulator(),
        "actions": StatsAccumulator(),
    }
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            state_values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action_values = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            accumulators["state"].update(state_values)
            accumulators["actions"].update(action_values)
    return {key: acc.finalize() for key, acc in accumulators.items()}


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.inputs if path]
    if not input_dirs:
        raise RuntimeError("No input datasets were provided")
    for directory in input_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Input dataset not found: {directory}")

    output_root = args.output_dir.resolve()
    ensure_dir(output_root, overwrite=args.overwrite)
    (output_root / DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    (output_root / VIDEOS_SUBDIR).mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (output_root / "images").mkdir(parents=True, exist_ok=True)
    meta_dir = output_root / METADATA_DIRNAME
    meta_dir.mkdir(parents=True, exist_ok=True)

    template_info: dict | None = None
    chunk_size = args.chunk_size
    camera_ids = args.camera_ids
    tasks_map: dict[str, int] = {}
    merged_episodes: list[EpisodeRecord] = []
    total_frames = 0

    for dataset_root in input_dirs:
        info, tasks, episodes = load_dataset(dataset_root)
        if template_info is None:
            template_info = copy.deepcopy(info)
            if chunk_size is None:
                chunk_size = info["chunks_size"]
            if camera_ids is None:
                camera_ids = detect_camera_ids(info)
        else:
            for key in ("robot_type", "fps", "data_path", "video_path"):
                if template_info[key] != info[key]:
                    raise ValueError(f"Dataset {dataset_root} mismatched metadata field '{key}'")
            if chunk_size is None:
                chunk_size = info["chunks_size"]
            elif chunk_size != info["chunks_size"]:
                raise ValueError(f"Chunk size mismatch in dataset {dataset_root}")

        for record in tasks:
            task_text = record["task"]
            tasks_map.setdefault(task_text, len(tasks_map))

        episodes_sorted = sorted(episodes, key=lambda rec: rec["episode_index"])
        for entry in tqdm(episodes_sorted, desc=f"Copy {dataset_root.name}", unit="episode"):
            local_idx = entry["episode_index"]
            global_idx = len(merged_episodes)
            copy_episode(
                source_root=dataset_root,
                target_root=output_root,
                local_idx=local_idx,
                global_idx=global_idx,
                chunk_size=chunk_size,
                camera_ids=camera_ids,
                copy_images=args.copy_images,
            )
            merged_episodes.append(
                EpisodeRecord(
                    episode_index=global_idx,
                    tasks=entry.get("tasks", []),
                    length=entry.get("length", 0),
                )
            )
            total_frames += entry.get("length", 0)

    if template_info is None or chunk_size is None or camera_ids is None:
        raise RuntimeError("Failed to initialize metadata from input datasets")

    total_episodes = len(merged_episodes)
    total_chunks = (total_episodes + chunk_size - 1) // chunk_size if total_episodes else 0
    template_info["total_episodes"] = total_episodes
    template_info["total_frames"] = total_frames
    template_info["total_tasks"] = len(tasks_map)
    template_info["total_videos"] = total_episodes * len(camera_ids)
    template_info["total_chunks"] = max(total_chunks, 1)
    template_info["splits"] = {"train": f"0:{total_episodes}"}

    features = template_info.get("features", {}).copy()
    for key in list(features):
        if key.startswith("observation.camera_") and key.endswith(".rgb"):
            cam_id = int(key.split("_")[1].split(".")[0])
            if cam_id not in camera_ids:
                del features[key]
    template_info["features"] = features

    episodes_payload = [record.__dict__ for record in merged_episodes]
    write_jsonl(meta_dir / "episodes.jsonl", episodes_payload)
    tasks_payload = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(tasks_map.items(), key=lambda item: item[1])
    ]
    write_jsonl(meta_dir / "tasks.jsonl", tasks_payload)
    with (meta_dir / "info.json").open("w", encoding="utf-8") as file:
        json.dump(template_info, file, ensure_ascii=False, indent=2)

    stats = compute_dataset_stats(output_root / DATA_SUBDIR)
    stats_path = output_root / STATS_REL_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    with (output_root / "norm_stats.json").open("w", encoding="utf-8") as file:
        json.dump({"norm_stats": stats}, file, ensure_ascii=False, indent=2)

    print("\nMerge complete")
    print(f"Inputs         : {[str(path) for path in input_dirs]}")
    print(f"Output         : {output_root}")
    print(f"Total episodes : {total_episodes}")
    print(f"Total frames   : {total_frames}")
    print(f"Stats written  : {stats_path}")


if __name__ == "__main__":
    main()
