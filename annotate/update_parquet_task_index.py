import json
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Update task_index in parquet files based on episodes.jsonl and tasks.jsonl")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset root (containing the 'meta' and 'data' folders)",
    )
    args = parser.parse_args()

    meta_dir = args.dataset_path / "meta"
    data_dir = args.dataset_path / "data"

    if not meta_dir.exists() or not data_dir.exists():
        raise FileNotFoundError(f"Invalid dataset path: {args.dataset_path}")

    # 1. Load tasks mapping
    with open("update_log.txt", "w") as log:
        log.write("Starting update script\n")
    
    tasks_path = meta_dir / "tasks.jsonl"
    print(f"Loading tasks from {tasks_path}")
    with open("update_log.txt", "a") as log:
        log.write(f"Loading tasks from {tasks_path}\n")
    task_to_id = {}
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            task_to_id[entry["task"]] = entry["task_index"]

    # 2. Load episodes mapping
    episodes_path = meta_dir / "episodes.jsonl"
    print(f"Loading episodes from {episodes_path}")
    episode_to_task_id = {}
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            # Assuming 'tasks' is a list of strings, and we take the first one
            # or the one that matches. LeRobot usually has one task per episode.
            tasks = entry.get("tasks", [])
            if isinstance(tasks, str):
                tasks = [tasks]
            
            if not tasks:
                print(f"Warning: Episode {entry['episode_index']} has no tasks.")
                continue
            
            task_desc = tasks[0]
            if task_desc not in task_to_id:
                print(f"Warning: Task '{task_desc}' for episode {entry['episode_index']} not found in tasks.jsonl")
                continue
            
            episode_to_task_id[entry["episode_index"]] = task_to_id[task_desc]

    # 3. Iterate over parquet files
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")

    updated_count = 0
    for p_file in tqdm(parquet_files):
        # Extract episode index from filename "episode_000123.parquet"
        try:
            episode_idx = int(p_file.stem.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name: {p_file}")
            continue

        if episode_idx not in episode_to_task_id:
            print(f"Warning: Episode {episode_idx} not found in episodes.jsonl")
            continue

        correct_task_index = episode_to_task_id[episode_idx]

        # Read parquet
        table = pq.read_table(p_file)
        
        # Check if update is needed
        if "task_index" in table.column_names:
            current_indices = table["task_index"].to_pylist()
            # Check if all are correct (usually constant per episode)
            if all(idx == correct_task_index for idx in current_indices):
                continue
        
        # Update task_index column
        # Create a new column with the correct index repeated for all rows
        num_rows = table.num_rows
        new_task_indices = np.full(num_rows, correct_task_index, dtype=np.int64)
        
        # Replace or append the column
        if "task_index" in table.column_names:
            # Drop existing and add new
            col_index = table.column_names.index("task_index")
            table = table.remove_column(col_index)
            
        # Add new column
        table = table.append_column("task_index", [new_task_indices])

        # Write back
        pq.write_table(table, p_file)
        updated_count += 1

    print(f"Updated {updated_count} parquet files.")

if __name__ == "__main__":
    main()
