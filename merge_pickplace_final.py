#!/usr/bin/env python3
"""Merge PickPlaceBottle shards into a single LeRobot dataset.

This script copies parquet files and encoded videos for every shard, rebuilds the
metadata (episodes.jsonl/info.json/tasks.jsonl), and recomputes normalization
statistics directly from the merged parquet files. The resulting dataset is fully
self-contained and does not require any post-processing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_ROOT = Path("/mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle")
STATS_PATH = Path("meta") / "stats.json"
EPISODES_FILE = "meta/episodes.jsonl"
TASKS_FILE = "meta/tasks.jsonl"
INFO_FILE = "meta/info.json"
DATA_SUBDIR = "data"
VIDEOS_SUBDIR = "videos"


@dataclass
class EpisodeRecord:
    episode_index: int
    tasks: list[str]
    length: int


class StatsAccumulator:
    """Streaming mean/std tracker that also stores samples for quantiles."""

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
        assert self.mean is not None and self.m2 is not None
        batch_count = array.shape[0]
        batch_mean = array.mean(axis=0)
        centered = array - batch_mean
        batch_m2 = (centered * centered).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            self.m2 = self.m2 + batch_m2 + (delta * delta) * self.count * batch_count / total
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
    parser = argparse.ArgumentParser(description="Merge PickPlaceBottle shards into one dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory that contains PickPlaceBottle_* shards",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Specific shard folder names to merge (relative to --root). If omitted, all PickPlaceBottle_<n> folders are used.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="PickPlaceBottle_Merged_v3",
        help="Name of the merged dataset folder that will be created under --root.",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Camera ids whose videos must be copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder if it already exists.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy preview images if they are present.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size. Defaults to the value found in the first shard.",
    )
    return parser.parse_args()


def discover_sources(root: Path, explicit: Sequence[str] | None) -> list[Path]:
    if explicit:
        return [root / name for name in explicit]
    pattern = re.compile(r"PickPlaceBottle_[0-9]+")
    return sorted([path for path in root.iterdir() if path.is_dir() and pattern.fullmatch(path.name)])


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


def load_template_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    
    # Read source parquet, update episode_index, and write to destination
    table = pq.read_table(src_data)
    if "episode_index" in table.column_names:
        table = table.remove_column(table.column_names.index("episode_index"))
    table = table.append_column("episode_index", [np.full(table.num_rows, global_idx, dtype=np.int64)])
    pq.write_table(table, dst_data_dir / f"episode_{global_idx:06d}.parquet")

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
                shutil.copy2(src_img, dst_img_dir / src_img.name.replace(f"{local_idx:06d}", f"{global_idx:06d}"))


def compute_dataset_stats(data_root: Path) -> dict[str, dict[str, list[float]]]:
    accumulators = {
        "state": StatsAccumulator(),
        "actions": StatsAccumulator(),
    }
    chunk_dirs = sorted(data_root.glob("chunk-*"))
    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for parquet_path in parquet_files:
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            state_values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action_values = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            accumulators["state"].update(state_values)
            accumulators["actions"].update(action_values)
    return {key: acc.finalize() for key, acc in accumulators.items()}


def main() -> None:
    args = parse_args()
    source_dirs = discover_sources(args.root, args.sources)
    if not source_dirs:
        raise RuntimeError(f"No source datasets found under {args.root}")

    output_root = (args.root / args.output_name).resolve()
    ensure_dir(output_root, overwrite=args.overwrite)
    (output_root / DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    (output_root / VIDEOS_SUBDIR).mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (output_root / "images").mkdir(parents=True, exist_ok=True)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    template_info: dict | None = None
    chunk_size = args.chunk_size
    tasks_map: dict[int, str] = {}
    merged_episodes: list[EpisodeRecord] = []
    total_frames = 0

    for src in source_dirs:
        meta_path = src / "meta"
        info_path = meta_path / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing info.json in {meta_path}")
        info = load_template_info(info_path)
        if template_info is None:
            template_info = info
            if chunk_size is None:
                chunk_size = info["chunks_size"]
        else:
            for key in ("robot_type", "fps", "data_path", "video_path"):
                if template_info[key] != info[key]:
                    raise ValueError(f"Shard {src.name} has mismatched {key}")
            if chunk_size is None:
                chunk_size = info["chunks_size"]
            elif chunk_size != info["chunks_size"]:
                raise ValueError(f"Chunk size mismatch in {src.name}")
        if chunk_size is None:
            raise RuntimeError("Unable to determine chunk size")

        tasks = read_jsonl(meta_path / "tasks.jsonl")
        for record in tasks:
            idx = record["task_index"]
            task_text = record["task"]
            if idx in tasks_map and tasks_map[idx] != task_text:
                raise ValueError(f"Conflicting task index {idx} between shards")
            tasks_map.setdefault(idx, task_text)

        episodes = read_jsonl(meta_path / "episodes.jsonl")
        episodes_sorted = sorted(episodes, key=lambda rec: rec["episode_index"])
        for entry in tqdm(episodes_sorted, desc=f"Copy {src.name}", unit="episode"):
            local_idx = entry["episode_index"]
            global_idx = len(merged_episodes)
            copy_episode(
                source_root=src,
                target_root=output_root,
                local_idx=local_idx,
                global_idx=global_idx,
                chunk_size=chunk_size,
                camera_ids=args.camera_ids,
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

    total_episodes = len(merged_episodes)
    total_chunks = (total_episodes + chunk_size - 1) // chunk_size if total_episodes else 0
    if template_info is None:
        raise RuntimeError("No template info loaded")

    template_info["total_episodes"] = total_episodes
    template_info["total_frames"] = total_frames
    template_info["total_tasks"] = len(tasks_map)
    template_info["total_videos"] = total_episodes * len(args.camera_ids)
    template_info["total_chunks"] = max(total_chunks, 1)
    template_info["splits"] = {"train": f"0:{total_episodes}"}

    features = template_info.get("features", {})
    for key in list(features):
        if key.startswith("observation.camera_") and key.endswith(".rgb"):
            cam_id = int(key.split("_")[1].split(".")[0])
            if cam_id not in args.camera_ids:
                features.pop(key, None)
    template_info["features"] = features

    episodes_payload = [record.__dict__ for record in merged_episodes]
    write_jsonl(meta_dir / "episodes.jsonl", episodes_payload)
    task_records = [
        {"task_index": idx, "task": task}
        for idx, task in sorted(tasks_map.items(), key=lambda item: item[0])
    ]
    write_jsonl(meta_dir / "tasks.jsonl", task_records)
    with (meta_dir / "info.json").open("w", encoding="utf-8") as file:
        json.dump(template_info, file, ensure_ascii=False, indent=2)

    stats = compute_dataset_stats(output_root / DATA_SUBDIR)
    stats_path = output_root / STATS_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    with (output_root / "norm_stats.json").open("w", encoding="utf-8") as file:
        json.dump({"norm_stats": stats}, file, ensure_ascii=False, indent=2)

    print("\nMerge complete")
    print(f"Total episodes  : {total_episodes}")
    print(f"Total frames    : {total_frames}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
