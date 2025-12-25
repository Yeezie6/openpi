import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging
import queue
import time
import numpy as np
import random
import glob

import decord
import sys
from PIL import Image
import base64

# OpenAI SDK (v1.x). If not installed, add to requirements.txt: openai>=1.51.0
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Will raise in init with helpful message

from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# New System Prompt
SYS_PROMPT = """
You are a video annotation AI specialized in identifying high-level manipulation tasks performed by a real robotic dexterous hand (not a human hand) in egocentric or third-person videos.

Input:
You will receive a sequence of video frames sampled every 0.5 seconds, showing a physical robotic dexterous hand interacting with objects in the real world.

Task:
Generate a single-sentence task description that summarizes the main manipulation action performed by the robotic hand in the video.

The task description MUST include:
- Which robotic hand is used (left robotic hand, right robotic hand, or both robotic hands).
- The grasp taxonomy used by the robotic hand (e.g., extension grasp, power grasp, precision grasp).
- The object being manipulated, with at least one distinctive feature (e.g., color, shape, position, or material).
- The specific part of the object that is grasped or contacted by the robotic hand (e.g., handle, edge, stem, lid, surface).

Guidelines:
- Use one sentence only.
- Use clear, concise, and concrete language suitable for robotic manipulation tasks.
- Use imperative or task-description style (e.g., “Use the right robotic hand to…”).
- Assume the manipulator is a real, physical dexterous robotic hand, not a human hand.
- Do not describe human anatomy (e.g., palm, fingernail, skin).
- If no meaningful robotic hand–object interaction is present, return null.

Output format:
Return plain text only.
Do not include explanations, JSON, or additional fields.
"""


def process_task_log_file(task_log_path: str) -> List[Dict[str, Any]]:
    # Modified to scan directory instead of reading task log json
    # task_log_path is treated as the directory to scan
    video_dir = task_log_path
    task_list = []
    
    # Support both mp4 and avi, or just mp4 as per request context
    files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not files:
        logger.warning(f"No .mp4 files found in {video_dir}")
    
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        task_list.append({
            "file_path": file_path,
            "part_id": "PickPlaceBottle",
            "task_name": "robot_manipulation",
            "video_idx": base_name,
            "base_name": base_name,
            "valid_segments": None # Will be determined in extract_frames
        })

    logger.info(f"Processed {len(task_list)} video entries from directory {video_dir}")
    return task_list


def extract_frames(video_path: str, extract_fps: int, valid_segment_list: Optional[List[int]], cached_chunk_list: List[int] = []) -> List[Image.Image]:
    try:
        vr = decord.VideoReader(video_path)
        local_fps = vr.get_avg_fps()
        if local_fps <= 0:
            local_fps = 20 # Fallback
            
        duration_sec = len(vr) / local_fps
        
        # If no segments provided, use the whole video
        if not valid_segment_list:
            valid_segment_list = [0, int(duration_sec)]
            
        segment_gaps = [end - start for start, end in zip(valid_segment_list[:-1], valid_segment_list[1:])]
        current_start = valid_segment_list[0]
        chunk_list = []
        images_list = []
        
        # If valid_segment_list has only 2 elements (start, end), loop won't run for gaps
        # We need to handle the segments properly.
        # If valid_segment_list is [0, 10], we want one chunk (0, 10).
        
        if len(valid_segment_list) == 2:
             chunk_list.append((valid_segment_list[0], valid_segment_list[1]))
        else:
            for i, gap in enumerate(segment_gaps):
                if gap > 1:
                    chunk_list.append((current_start, valid_segment_list[i] + 1))
                    current_start = valid_segment_list[i + 1]
            chunk_list.append((current_start, valid_segment_list[-1] + 1))

        # Split large chunks (simple logic from original code)
        filterd_chunk_list = []
        for start, end in chunk_list:
            if (end - start) > 20: # Increased threshold slightly as we might have longer videos
                mid = (start + end) // 2
                filterd_chunk_list.append((start, mid))
                filterd_chunk_list.append((mid, end))
            else:
                filterd_chunk_list.append((start, end))
        chunk_list = filterd_chunk_list

        unprocessed_chunk_list = []
        for chunk_id, (start, end) in enumerate(chunk_list):
            if start in cached_chunk_list:
                logger.info(f"Skipping chunk {chunk_id} as it is in the cached_chunk_list")
                continue
            
            start_frame_idx = int(start * local_fps)
            end_frame_idx = int(end * local_fps)
            
            # Ensure indices are within bounds
            start_frame_idx = max(0, min(start_frame_idx, len(vr)-1))
            end_frame_idx = max(0, min(end_frame_idx, len(vr)))
            
            if start_frame_idx >= end_frame_idx:
                continue

            extract_frame_num = int((end - start) * extract_fps) + 1
            frame_indexes = np.linspace(start_frame_idx, end_frame_idx - 1, extract_frame_num, dtype=int).tolist()
            
            try:
                frames = vr.get_batch(frame_indexes).asnumpy()
                chunk_images = []
                for frame in frames:
                    pil_image = Image.fromarray(frame)
                    chunk_images.append(pil_image)
                images_list.append(chunk_images)
                unprocessed_chunk_list.append((start, end))
            except Exception as e:
                logger.error(f"Error getting batch for {video_path}: {e}")
                continue
                
        logger.debug(f"Extracted {len(images_list)} chunks of frames from {video_path}")
        return images_list, unprocessed_chunk_list
    except Exception as e:
        logger.error(f"Failed to extract frames from {video_path}: {e}")
        return [], []


def pil_to_base64_png(img: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class OpenAIAnalyzer:
    """Handle OpenAI API interactions (multi-modal chat)."""

    def __init__(self, api_key: str, model: str):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. Please `pip install openai>=1.51.0`.")
        # Init client with custom base URL
        self.client = OpenAI(api_key=api_key, base_url="https://api.gptoai.top/v1")
        self.model = model
        logger.info("OpenAI client initialized successfully")

    def analyze_hand_movements(self, frames: List[Image.Image], task_name: str) -> Dict[str, Any]:
        try:
            prompt = (
                f"Analyze the video frames in the scene. Here we provide {len(frames)} "
                f"frames sampled from a video of {(len(frames) - 1) // 2} seconds. "
                "Follow the instruction to analyze the hand movements and interactions."
            )

            # Build content: system + user(text+images)
            user_content = [{"type": "text", "text": prompt}]
            for img in frames:
                b64_url = pil_to_base64_png(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_url}
                })

            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_content},
            ]

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    time.sleep(random.uniform(1, 3))
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=0
                    )
                    # Extract text
                    response_text = resp.choices[0].message.content.strip()
                    
                    # Return simple dict with text
                    return {"task_description": response_text}

                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed, retrying: {e}")
                    sleep_time = random.uniform(5, 30) * (attempt + 1)
                    time.sleep(sleep_time)

            logger.error("OpenAI API failed after multiple attempts")
            return {}
        except Exception as e:
            logger.error(f"Failed to analyze with OpenAI: {e}")
            return {}


class ResultAggregator:
    @staticmethod
    def find_exist_files(json_dir: str) -> Dict[str, List[int]]:
        existing_files = {}
        if not os.path.exists(json_dir):
            return existing_files
            
        for jsonl_file in os.listdir(json_dir):
            if jsonl_file.endswith(".jsonl"):
                with open(os.path.join(json_dir, jsonl_file), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            if 'error' in result:
                                continue
                            if result.get("base_name"):
                                if result["base_name"] not in existing_files:
                                    existing_files[result["base_name"]] = []
                                if "start_second" in result:
                                    existing_files[result["base_name"]].append(int(result['start_second']))
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON line from {jsonl_file}: {e}")
        logger.info(f"Found {len(existing_files)} existing video annotations in {json_dir}")
        return existing_files

    @staticmethod
    def aggregate_results(json_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for jsonl_file in os.listdir(json_dir):
            if jsonl_file.endswith(".jsonl"):
                with open(os.path.join(json_dir, jsonl_file), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            if result.get("error"):
                                logger.warning(f"Skipping result with error:{result.get('base_name')} - {result['error']}")
                                continue
                            if result.get("base_name"):
                                grouped[result["base_name"]].append(result)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON line from {jsonl_file}: {e}")

        final_results = {}
        for base_name, chunks in grouped.items():
            # Sort by start_second if available
            chunks.sort(key=lambda x: int(x.get("start_second", -1)))
            final_results[base_name] = chunks

        return final_results

    @staticmethod
    def save_results(results: Dict[str, Any], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, openai_api_key: str, model: str, worker_output_path: str, worker_id: int):
    logger.info(f"Worker {worker_id} started")

    analyzer = OpenAIAnalyzer(openai_api_key, model)
    while True:
        try:
            (task, processed_chunk_list) = task_queue.get(timeout=5)
            if task is None:
                logger.info(f"Worker {worker_id} received stop signal")
                break

            file_metadata = task
            logger.info(f"Worker {worker_id} processing: {file_metadata['file_path']}")

            frames, unprocessed_chunk_list = extract_frames(
                file_metadata["file_path"], extract_fps=2,
                valid_segment_list=file_metadata["valid_segments"],
                cached_chunk_list=processed_chunk_list
            )
            logger.info(f"{len(frames)} chunks of frames extracted from {file_metadata['file_path']} with {len(processed_chunk_list)} chunks skipped")
            result = 0
            for chunked_frames, (start, end) in zip(frames, unprocessed_chunk_list):
                logger.info(f"Chunk from {start} to {end} seconds has {len(chunked_frames)} frames")
                if len(chunked_frames) < 3:
                    logger.warning(f"Skipping chunk from {start} to {end} seconds due to insufficient frames")
                    continue

                annotations = analyzer.analyze_hand_movements(chunked_frames, file_metadata["task_name"])
                if annotations:
                    annotations["base_name"] = file_metadata["base_name"]
                    annotations["start_second"] = start
                    annotations["end_second"] = end

                    with open(worker_output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(annotations, ensure_ascii=False) + '\n')
                    result += 1
                else:
                    with open(worker_output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "base_name": file_metadata["base_name"],
                            "start_second": start,
                            "error": "No annotations returned from OpenAI"
                        }, ensure_ascii=False) + '\n')

            result_queue.put((result, len(frames)))

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} error processing {file_metadata['file_path']}: {e}")
            result_queue.put((0, 0))

    logger.info(f"Worker {worker_id} finished")


class VideoAnnotationPipeline:
    def __init__(self, task_log_file_path: str, openai_api_key_list: List[str], model: str, num_processes: int = 4, node_id: int = 0, node_num: int = 1):
        # task_log_file_path is now the directory path
        self.task_log_file_path = task_log_file_path
        self.result_aggregator = ResultAggregator()
        self.openai_api_key_list = openai_api_key_list
        self.model = model
        self.num_processes = num_processes
        self.node_id = node_id
        self.node_num = node_num

    def run(self, output_file: str, output_dir: str = "./video_anno") -> Dict[str, Any]:
        logger.info("Discovering video files...")
        video_files = process_task_log_file(self.task_log_file_path)
        os.makedirs(output_dir, exist_ok=True)

        if not video_files:
            logger.warning("No video files found!")
            return {"total_base_names": 0, "results": {}}

        local_video_files = video_files[self.node_id::self.node_num]
        processed_files = self.result_aggregator.find_exist_files(output_dir)

        args_list = []
        for vf in local_video_files:
            if vf["base_name"] in processed_files:
                args_list.append((vf, processed_files[vf["base_name"]]))
            else:
                args_list.append((vf, []))

        task_queue = mp.Queue()
        result_queue = mp.Queue()
        for file_meta in args_list:
            task_queue.put(file_meta)

        start_time = time.time()
        logger.info(f"Starting {self.num_processes} worker processes...")
        processes = []
        for i in range(self.num_processes):
            p = mp.Process(
                target=worker_process,
                args=(task_queue, result_queue, self.openai_api_key_list[i % len(self.openai_api_key_list)], self.model,
                      os.path.join(output_dir, f"worker_output_{i + self.node_id * self.num_processes}.jsonl"), i + 1 + self.node_id * self.num_processes)
            )
            p.start()
            processes.append(p)

        total_tasks = len(local_video_files)
        completed = 0
        successful = 0
        half_success = 0
        success_clip = 0
        total_clips = 0

        logger.info(f"Processing {total_tasks} videos...")
        while completed < total_tasks:
            try:
                (result, target_result) = result_queue.get(timeout=1800)
                completed += 1
                if result == target_result and target_result > 0:
                    successful += 1
                elif result > 0:
                    half_success += 1

                total_clips += target_result
                success_clip += result
                current_time = time.time()
                if completed > 0:
                    logger.info(
                        f"Success {successful} + Half Success {half_success} of {completed} from {total_tasks} total videos, Consumed: {int(current_time - start_time)} seconds, Remaining: {int((total_tasks - completed) * (current_time - start_time) / completed)} seconds, Progress: {completed}/{total_tasks} ({(completed / total_tasks) * 100:.2f}%)"
                    )
            except queue.Empty:
                logger.warning("Timeout waiting for results")
                break

        for _ in range(self.num_processes):
            task_queue.put((None, []))

        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                logger.warning(f"Force terminating worker process {p.pid}")
                p.terminate()

        logger.info("All workers finished")

        if self.node_id == 0:
            logger.info("Aggregating results...")
            aggregated_results = self.result_aggregator.aggregate_results(output_dir)
            final_output = {
                "total_base_names": len(aggregated_results),
                "results": aggregated_results
            }
            self.result_aggregator.save_results(final_output, output_file)


def main():
    node_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    node_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    api_key_st = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    api_key_ed_raw = sys.argv[4] if len(sys.argv) > 4 else "-1"
    try:
        api_key_ed = int(api_key_ed_raw)
    except ValueError:
        api_key_ed = -1

    # Configuration
    # Provide your OpenAI API keys list via env or inline
    OPENAI_API_KEYS = ["sk-aPCXdOLYxqjFbNKwvh4JeqKbTLpTs30mY2MvjvKcoK78uDzm"]  # TODO: replace with real keys or load from env

    # Treat -1 as "until end"
    if api_key_ed == -1:
        api_key_list = OPENAI_API_KEYS[api_key_st:]
    else:
        api_key_list = OPENAI_API_KEYS[api_key_st:api_key_ed]

    if not api_key_list:
        raise ValueError("No OpenAI API keys selected. Adjust start/end indices or provide keys.")

    # TODO: Confirm the correct multi-modal model name (e.g., "gpt-4o-mini" / "gpt-4o")
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"Using {len(api_key_list)} API key(s) for OpenAI")
    print(f"Using model: {MODEL_NAME}")

    # Dynamically cap processes to avoid spawning excessive workers with few keys
    # For debugging network issues keep concurrency at 1
    NUM_PROCESSES = 1
    
    # Updated output paths to avoid conflict with original script
    OUTPUT_DIR = "/share/HaWoR/cc/openai_robot"
    OUTPUT_FILE = "/share/HaWoR/cc/openai_robot/annotations_robot.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-flight connectivity test (simple empty text request)
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK still not installed correctly.")
    try:
        test_client = OpenAI(api_key=api_key_list[0], base_url="https://api.gptoai.top/v1")
        _ = test_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            temperature=0,
        )
        logger.info("Pre-flight connectivity test succeeded.")
    except Exception as e:
        logger.error(f"Pre-flight connectivity test failed: {e}")
        logger.error("Abort before spawning workers. Please check network (firewall/DNS), model name或API Key。")
        return

    # Updated dataset path
    DATASET_PATH = "PickPlaceBottle/PickPlaceBottle_Merged_v2/videos/chunk-000"
    
    
    pipeline = VideoAnnotationPipeline(
        task_log_file_path=DATASET_PATH,
        openai_api_key_list=api_key_list,
        model=MODEL_NAME,
        num_processes=NUM_PROCESSES,
        node_id=node_id,
        node_num=node_num
    )

    try:
        pipeline.run(output_dir=OUTPUT_DIR, output_file=OUTPUT_FILE)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
