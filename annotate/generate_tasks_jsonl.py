#!/usr/bin/env python3
"""
Generates a tasks.jsonl file by extracting unique task descriptions from an existing episodes.jsonl file.
This ensures that the tasks.jsonl file is consistent with the episodes.jsonl file for LeRobot datasets.

Usage:
    python annotate/generate_tasks_jsonl.py --dataset-path /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_MegaMerge
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate tasks.jsonl from episodes.jsonl")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset root (containing the 'meta' folder)",
    )
    args = parser.parse_args()

    meta_dir = args.dataset_path / "meta"
    if not meta_dir.exists():
        # Try checking if the path itself is the meta dir
        if args.dataset_path.name == "meta":
            meta_dir = args.dataset_path
        else:
            raise FileNotFoundError(f"Could not find 'meta' directory in {args.dataset_path}")

    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")

    print(f"Reading episodes from: {episodes_path}")
    
    unique_tasks = []
    task_to_index = {}

    # Read episodes and collect unique tasks
    with episodes_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tasks = record.get("tasks", [])
                # Handle case where tasks might be a single string instead of list (though spec says list)
                if isinstance(tasks, str):
                    tasks = [tasks]
                
                for task in tasks:
                    if task not in task_to_index:
                        idx = len(unique_tasks)
                        task_to_index[task] = idx
                        unique_tasks.append(task)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")

    print(f"Found {len(unique_tasks)} unique tasks.")

    # Write tasks.jsonl
    print(f"Writing tasks to: {tasks_path}")
    with tasks_path.open("w", encoding="utf-8") as f:
        for idx, task in enumerate(unique_tasks):
            entry = {"task_index": idx, "task": task}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
