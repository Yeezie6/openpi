"""
Minimal example script for converting a local LIBERO dataset to LeRobot format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "yiqing/libero_local"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def load_hdf5_dataset(dataset_dir: Path):
    """Load episodes from HDF5 files in a dataset directory."""
    episodes = []
    hdf5_files = sorted(dataset_dir.glob("*.hdf5"))
    
    print(f"  Found {len(hdf5_files)} HDF5 files")
    
    for hdf5_file in hdf5_files:
        # Extract task name from filename
        task_name = hdf5_file.stem.replace("_demo", "").replace("_", " ")
        
        with h5py.File(hdf5_file, "r") as f:
            # Each file contains multiple demos under 'data/demo_X'
            demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            
            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                
                # Extract data
                actions = np.array(demo["actions"])
                robot_states = np.array(demo["robot_states"])
                agentview_images = np.array(demo["obs"]["agentview_rgb"])
                wrist_images = np.array(demo["obs"]["eye_in_hand_rgb"])
                
                episodes.append({
                    "task": task_name,
                    "actions": actions,
                    "robot_states": robot_states,
                    "agentview_images": agentview_images,
                    "wrist_images": wrist_images,
                })
    
    return episodes


def main(data_dir: str, *, push_to_hub: bool = False):
    data_dir = Path(data_dir)
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `actions`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),  # robot_states has 9 dimensions
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    total_episodes = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        dataset_path = data_dir / raw_dataset_name
        print(f"\nProcessing dataset: {raw_dataset_name} from {dataset_path}")
        
        if not dataset_path.exists():
            print(f"  Warning: {dataset_path} does not exist, skipping...")
            continue
        
        # Load episodes from HDF5 files
        episodes = load_hdf5_dataset(dataset_path)
        print(f"  Loaded {len(episodes)} episodes")
        
        # Write episodes to LeRobot dataset
        for ep_idx, episode in enumerate(episodes):
            num_frames = len(episode["actions"])
            
            for frame_idx in range(num_frames):
                # Resize images from 128x128 to 256x256
                image = cv2.resize(
                    episode["agentview_images"][frame_idx],
                    (256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                wrist_image = cv2.resize(
                    episode["wrist_images"][frame_idx],
                    (256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": episode["robot_states"][frame_idx].astype(np.float32),
                        "actions": episode["actions"][frame_idx].astype(np.float32),
                        "task": episode["task"],
                    }
                )
            
            dataset.save_episode()
            total_episodes += 1
            
            if (ep_idx + 1) % 10 == 0:
                print(f"  Processed {ep_idx + 1}/{len(episodes)} episodes")
    
    print(f"\nTotal episodes processed: {total_episodes}")
    print(f"Dataset saved to: {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "local"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
