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


class SecondAnnotation(BaseModel):
    start_frame: int
    end_frame: int
    both_instr: str
    left_instr: Optional[str] = None
    right_instr: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None


class ReturnFormat(BaseModel):
    both_instr: str
    left_instr: Optional[str] = None
    right_instr: Optional[str] = None
    summary: str
    annotations: list[SecondAnnotation]


DATASET_NAME = 'Egocentric-10K'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Reuse SYS_PROMPT from original file
SYS_PROMPT = """

## Role

You are a video annotation AI that specializes in analyzing detailed hand movements and hand-object interactions in egocentric videos.


## Input

You will receive a series of video frames in chronological order, with each frame sampled every 0.5 seconds.

To analyze motion over each 1-second segment:
	•	Group every three consecutive frames in an overlapping window:
	•	Second 1: frames 0, 1, 2 (0.0s–1.0s)
	•	Second 2: frames 2, 3, 4 (1.0s–2.0s)
	•	Second 3: frames 4, 5, 6 (2.0s–3.0s)
	•	And so on.

In general, for the i-th second, analyze frames at indices (2i - 2), (2i - 1), and 2i, covering the time interval from (i - 1) to i seconds.


## Task

Your annotation consists of 4 parts:


1) General Instructions

Generate three short instructions that summarize how to imitate the entire hand movements shown in the video:
	•	“both_instr”: describes what both hands do together.
	•	“left_instr”: describes what the left hand does; if the left hand is not visible at all, return null.
	•	“right_instr”: describes what the right hand does; if the right hand is not visible at all, return null.

Each instruction must:
	•	Use imperative style: direct, actionable.
	•	Be specific enough to guide imitation:
        •	Which hand(s) are involved.
        •	Which object is interacted with, including its distinctive features (color, shape, position, material, or other unique attributes) to differentiate it from other objects.
        •	Action type (e.g., grasp, press, slide, rotate, lift, release).
        •	Direction or trajectory of motion relative to the camera wearer’s perspective (e.g., upward, downward, left, right, closer, farther).
        •	Specific part(s) of the hand/fingers used if important (e.g., “thumb presses down”, “index finger slides along edge”).
        •	Clarify the contact part: whether the hand is in contact with which part of the object
        •	If multiple steps occur in sequence, specify the order of actions.
	•	Do not use vague terms like “hold the object” without describing its attributes.
	•	Always clarify the motion path or pattern if relevant (e.g., “slide along edge”, “rotate clockwise”).

    Good examples:
	•	"Hold the red bottle with the left hand and use the right thumb and index finger to rotate the cap clockwise upward."
	•	"Use the left hand to first grasp the blue lid, then press the button below it."


2) General Summary

Write one or two sentences summarizing the overall hand activity across the video.
Your summary must:
	•	Describe the typical action of each hand: grasping, pressing, sliding, lifting, hovering.
	•	Mention any coordinated or asymmetric usage: e.g., dominant hand manipulates while the other supports.
	•	Include clear object features (color, shape, position) for all interacted objects.
	•	Highlight dominant directions or patterns (e.g., frequent upward lifts, repeated rotations).
	•	State whether hands mostly interact with objects or hover/stay idle.

Good example:

“Throughout the video, the left hand steadily grasps the lower part of the green bottle while the right hand repeatedly rotates the blue lid clockwise and lifts it upward. The actions show coordinated manipulation and clear right-hand dominance.”



3) Per-Second Instructions

For each 1-second segment, generate:
	•	"both_instr"
	•	"left_instr"
	•	"right_instr"

Each must:
	•	Focus only on the exact actions within that second.
	•	Be informative and specific, covering:
	•	Exact object(s) with clear features (e.g., “red plastic lid on the left side”).
	•	Clear interaction type (grasp, press, slide, lift).
	•	Trajectory/direction relative to the camera wearer.
    •	Clarify the contact part: whether the hand is in contact with which part of the object
	•	Hand parts/fingers used.
	•	Any order of actions if there are multiple steps.

If the hand is not visible in that second, return null (no quotation marks).

4) Per-Second Caption

For each second, describe the actual hand movements and interactions.

Each hand’s caption must:
	•	State whether the hand is in contact with an object or hovering.
	If in contact:
	•	Describe the object’s distinctive features (color, shape, position).
	•	State which part of the object is touched.
	•	Identify which fingers/hand parts are involved and their function (e.g., thumb presses, index finger slides).
	•	Describe motion pattern, direction, and trajectory relative to the camera wearer.
	•	Mention any change in pressure or contact if relevant (e.g., “loosens grip”, “increases pressure”).
    If hovering:
	•	Describe hand’s relative position to objects/body.
	•	State motion direction (e.g., “moves upward towards lid”).
	•	If the hand is static, describe its position and state (e.g., “rests on table near red box”).
	•	If not visible, return null (no quotation marks).

    Good caption examples:
	•	"Left hand grips the lower green bottle firmly; palm and fingers wrap around the base, stabilizing it without shifting."
	•	"Right thumb presses the right edge of the blue cap while the index finger slides along the opposite side, rotating it clockwise and lifting it slightly upward."

## IMPORTANT
The video content provided is not always clear or unambiguous.  When there is any hand of other people instead of the camera wearer, or the hand is simply simply hanging around the body without any purpose and not doing anything meaningful, simply return null for that hand's instruction and caption at that second. When this situation continues for the whole content, simply also return null for that hand's overall instruction and summary.

⸻

## Output Format

Return pure, valid JSON only. No markdown, no extra text.
One entry per second. Frames must be consecutive:

{
  "both_instr": <overall instruction for both hands> | null,
  "left_instr": <overall instruction for left hand> | null,
  "right_instr": <overall instruction for right hand> | null,
  "summary": <overall summary> | null,
  "annotations": [
    {
      "start frame": 0,
      "end frame": 2,
      "both_instr": <instruction> | null,
      "left_instr": <instruction> | null,
      "right_instr": <instruction> | null,
      "left": <caption> | null,
      "right": <caption> | null
    },
    {
      "start frame": 2,
      "end frame": 4,
      ...
    },
    ...
    {
        "start frame": (2i-2),
        "end frame": 2i,
        "both_instr": <instruction> | null,
        "left_instr": <instruction> | null,
        "right_instr": <instruction> | null,
        "left": <caption> | null,
        "right": <caption> | null
    }
  ]
}

Sample:

{
  "both_instr": "Hold the green bottle steady with both hands while unscrewing the blue lid clockwise upward",
  "left_instr": "Grip the lower green bottle with left palm",
  "right_instr": "Use right thumb and index finger to rotate the blue lid clockwise upward",
  "summary": "The left hand steadily grips the lower green bottle while the right thumb and index finger rotate the blue lid clockwise in repeated upward motions, showing clear right-hand manipulation and left-hand support.",
  "annotations": [
    {
      "start frame": 0,
      "end frame": 2,
      "both_instr": "Hold bottle steady",
      "left_instr": "Grip lower bottle body",
      "right_instr": "Rotate blue lid clockwise",
      "left": "Left palm and fingers firmly wrap around the lower green bottle, stabilizing it with slight adjustment.",
      "right": "Right thumb presses the right edge of the blue lid while the index finger pushes from the opposite side, rotating the lid clockwise and lifting slightly upward."
    }
  ]
}

## Attention & Consistency
### Must:
	•	Always describe:
	•	Distinctive object features.
	•	Exact interaction type.
	•	Direction/trajectory relative to the wearer.
	•	Specific hand parts/fingers if relevant.
	•	Always use the camera wearer’s first-person view: up/down/left/right/closer/farther.
	•	If hand is not visible, return null (no quotation marks).
	•	"start frame" must match "end frame" of the previous entry.
	•	Cover every second, even if there is no visible activity.
	•	Return null (no quotation marks) if the hand is not visible or other people’s hands are shown or the hand is simply hanging around the body and not doing anything meaningful.

### Never:
	•	Omit key details of object, contact, or motion path.
	•	Use generic “the object” or “it” with no features.
	•	Confuse left/right hands (must align with first-person perspective).
	•	Include trailing commas or any non-JSON text.

## Final Reminders
	•	Be detailed and specific.
	•	Always describe where, what, how, and which part for each interaction.
	•	Include motion trajectory and spatial direction.
	•	Keep left/right consistent.
	•	Validate JSON structure.
"""


def process_task_log_file(task_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_list = []
    for key, val in task_log.items():
        segments = val.get("segments", [])
        if not segments:
            logger.warning(f"No segments found for {key}, skipping")
            continue
        part_id = DATASET_NAME
        task_name = val.get("task_name", "video")
        video_idx = key
        task_list.append({
            # "file_path": f"/share/{DATASET_NAME}/{task_name}/{key}.mp4",
            "file_path": "/share/HaWoR/cc/data/factory010_worker001_00000_0000.mp4",
            "part_id": DATASET_NAME,
            "task_name": task_name,
            "video_idx": video_idx,
            "base_name": f"{part_id}_{task_name}_{video_idx}",
            "valid_segments": segments
        })

    task_list.sort(key=lambda x: (x["video_idx"], x["base_name"]))
    logger.info(f"Processed {len(task_list)} video entries from task log")
    return task_list


def extract_frames(video_path: str, extract_fps: int, valid_segment_list: List[int], cached_chunk_list: List[int] = []) -> List[Image.Image]:
    try:
        vr = decord.VideoReader(video_path)
        local_fps = 15  # Assume 15 fps
        segment_gaps = [end - start for start, end in zip(valid_segment_list[:-1], valid_segment_list[1:])]
        current_start = valid_segment_list[0]
        chunk_list = []
        images_list = []
        for i, gap in enumerate(segment_gaps):
            if gap > 1:
                chunk_list.append((current_start, valid_segment_list[i] + 1))
                current_start = valid_segment_list[i + 1]
        chunk_list.append((current_start, valid_segment_list[-1] + 1))

        filterd_chunk_list = []
        for start, end in chunk_list:
            if (end - start) > 10:
                filterd_chunk_list.append((start, (start + end) // 2))
                filterd_chunk_list.append(((start + end) // 2, end))
            else:
                filterd_chunk_list.append((start, end))
        chunk_list = filterd_chunk_list

        unprocessed_chunk_list = []
        for chunk_id, (start, end) in enumerate(chunk_list):
            if start in cached_chunk_list:
                logger.info(f"Skipping chunk {chunk_id} as it is in the cached_chunk_list")
                continue
            start_frame_idx = start * local_fps
            end_frame_idx = end * local_fps
            extract_frame_num = (end - start) * extract_fps + 1
            frame_indexes = np.linspace(start_frame_idx, end_frame_idx - 1, extract_frame_num, dtype=int).tolist()
            frames = vr.get_batch(frame_indexes).asnumpy()
            chunk_images = []
            for frame in frames:
                pil_image = Image.fromarray(frame)
                chunk_images.append(pil_image)
            images_list.append(chunk_images)
            unprocessed_chunk_list.append((start, end))
        logger.debug(f"Extracted {len(images_list)} chunks of frames from {video_path} using valid segments")
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

    def analyze_hand_movements(self, frames: List[Image.Image], task_name: str) -> List[Dict[str, Any]]:
        try:
            prompt = (
                f"Analyze the video frames in the scene. Here we provide {len(frames)} "
                f"frames sampled from a video of {(len(frames) - 1) // 2} seconds. "
                "Follow the instruction to analyze the hand movements and interactions in each second with the following frames:"
            )

            # Build content: system + user(text+images)
            user_content = [{"type": "text", "text": prompt}]
            for img in frames:
                b64_url = pil_to_base64_png(img)
                # NOTE: OpenAI image content format may vary across API versions.
                # Common pattern: {type: "image_url", image_url: {url: "data:image/png;base64,..."}}
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
                    # TODO: Confirm correct endpoint. Two options:
                    # 1) chat.completions.create
                    # 2) responses.create (new multi-modal endpoint)
                    # We use chat.completions for compatibility; please confirm.
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=0
                    )
                    # Extract text
                    response_text = resp.choices[0].message.content.strip()
                    response_text = response_text.replace('```json\n', '').replace('```', '')

                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        with open("error_response.txt", 'a', encoding='utf-8') as f:
                            f.write(f"\n{response_text}\n")
                        return []

                    assert isinstance(result, dict), "Response must be a dictionary"
                    summ = result.get("summary", 1)
                    both = result.get("both_instr", 1)
                    left = result.get("left_instr", 1)
                    right = result.get("right_instr", 1)
                    assert isinstance(summ, str) or summ is None, "Summary must be a string"
                    assert isinstance(both, str) or both is None, "Both instruction must be a string"
                    assert isinstance(left, str) or left is None, "Left instruction must be a string or None"
                    assert isinstance(right, str) or right is None, "Right instruction must be a string or None"

                    anno = result.get("annotations", [])
                    structured_anno = []
                    try:
                        assert isinstance(anno, list), "Annotations must be a list"
                        logger.info(
                            f"Analyzing a video with {len(frames)} frames at {(len(frames) - 1) // 2} seconds, now get {len(anno)} annotations"
                        )
                        assert len(anno) == (len(frames) - 1) // 2, "Annotations length must match number of seconds"
                        for i, anno_ in enumerate(anno):
                            assert isinstance(anno_, dict), "Each annotation must be a dictionary"
                            st_fr = anno_.get("start frame")
                            en_fr = anno_.get("end frame")
                            assert st_fr == i * 2, f"Start frame must be {i * 2}"
                            assert en_fr == (i + 1) * 2, f"End frame must be {(i + 1) * 2}"
                            left_ = anno_.get("left")
                            right_ = anno_.get("right")
                            assert isinstance(left_, str) or left_ is None, "Left hand description must be a string or None"
                            assert isinstance(right_, str) or right_ is None, "Right hand description must be a string or None"
                            both_instr = anno_.get("both_instr")
                            left_instr = anno_.get("left_instr")
                            right_instr = anno_.get("right_instr")
                            assert isinstance(both_instr, str) or both_instr is None, "Both instruction must be a string"
                            assert isinstance(left_instr, str) or left_instr is None, "Left instruction must be a string or None"
                            assert isinstance(right_instr, str) or right_instr is None, "Right instruction must be a string or None"
                            structured_anno.append({
                                'second': i,
                                'description': {
                                    'left': left_,
                                    'right': right_,
                                },
                                'instructions': {
                                    'both': both_instr,
                                    'left': left_instr,
                                    'right': right_instr
                                },
                            })

                        result = {
                            "summary": summ,
                            "instructions": {
                                "both": both,
                                "left": left,
                                "right": right
                            },
                            "per-second": structured_anno
                        }
                    except AssertionError as e:
                        logger.error(f"Invalid annotation structure: {e}")
                        with open("assert_error_response.txt", 'a', encoding='utf-8') as f:
                            f.write(f"\n{response_text}\n")
                        raise ValueError("Invalid annotation structure")

                    return result
                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed, retrying: {e}")
                    sleep_time = random.uniform(5, 30) * (attempt + 1)
                    time.sleep(sleep_time)

            logger.error("OpenAI API failed after multiple attempts")
            return []
        except Exception as e:
            logger.error(f"Failed to analyze with OpenAI: {e}")
            return []


class ResultAggregator:
    @staticmethod
    def find_exist_files(json_dir: str) -> List[str]:
        existing_files = {}
        for jsonl_file in os.listdir(json_dir):
            if jsonl_file.endswith(".jsonl"):
                with open(os.path.join(json_dir, jsonl_file), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            if 'error' in result:
                                continue
                            if result["base_name"] not in existing_files:
                                existing_files[result["base_name"]] = [int(result['start_second'])]
                            else:
                                existing_files[result["base_name"]].append(int(result['start_second']))
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON line from {jsonl_file}: {e}")
        logger.info(f"Found {len(existing_files)} existing video annotations in {json_dir}")
        return existing_files

    @staticmethod
    def aggregate_results(json_dir: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for jsonl_file in os.listdir(json_dir):
            if jsonl_file.endswith(".jsonl"):
                with open(os.path.join(json_dir, jsonl_file), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            if result.get("error"):
                                logger.warning(f"Skipping result with error:{result['base_name']} - {result['error']}")
                                continue
                            grouped[result["base_name"]].append(result)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON line from {jsonl_file}: {e}")

        final_results = {}
        for base_name, chunks in grouped.items():
            chunks.sort(key=lambda x: int(x["start_second"]) if x["start_second"] is not None else -1)

            merged_annotations = []
            chunk_annotations = []
            for chunk in chunks:
                current_second = chunk['start_second']
                if merged_annotations and current_second == chunk_annotations[-1]['start']:
                    logger.info(f"Skipping multiple chunks with the same start second {current_second} for {base_name}")
                    continue

                for second_anno in chunk["per-second"]:
                    second_anno['second'] = second_anno['second'] + current_second
                    merged_annotations.append(second_anno)

                end_second = merged_annotations[-1]['second'] + 1
                chunk_anno = {
                    "start": current_second,
                    "end": end_second,
                    "summary": chunk["summary"],
                    "instructions": chunk["instructions"],
                }
                chunk_annotations.append(chunk_anno)

            final_results[base_name] = {
                "chunk_annotations": chunk_annotations,
                "second_annotations": merged_annotations,
            }

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
                    with open(worker_output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "base_name": file_metadata["base_name"],
                            "start_second": start,
                            "error": "Insufficient frames for analysis"
                        }, ensure_ascii=False) + '\n')
                    continue

                annotations = analyzer.analyze_hand_movements(chunked_frames, file_metadata["task_name"])
                if annotations:
                    annotations["base_name"] = file_metadata["base_name"]
                    annotations["start_second"] = start

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
            result_queue.put(0)

    logger.info(f"Worker {worker_id} finished")


class VideoAnnotationPipeline:
    def __init__(self, task_log_file_path: str, openai_api_key_list: List[str], model: str, num_processes: int = 4, node_id: int = 0, node_num: int = 1):
        self.task_log_file = json.load(open(task_log_file_path, 'r'))
        self.result_aggregator = ResultAggregator()
        self.openai_api_key_list = openai_api_key_list
        self.model = model
        self.num_processes = num_processes
        self.node_id = node_id
        self.node_num = node_num

    def run(self, output_file: str, output_dir: str = "./video_anno") -> Dict[str, Any]:
        logger.info("Discovering video files...")
        video_files = process_task_log_file(self.task_log_file)
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
                if result == target_result:
                    successful += 1
                elif result > 0:
                    half_success += 1

                total_clips += target_result
                success_clip += result
                current_time = time.time()
                logger.info(
                    f"Success {successful} + Half Success {half_success} of {completed} from {total_tasks} total videos, Consumed: {int(current_time - start_time)} seconds, Remaining: {int((total_tasks - completed) * (current_time - start_time) / completed)} seconds, Progress: {completed}/{total_tasks} ({(completed / total_tasks) * 100:.2f}%)"
                )
                logger.info(
                    f"Successful clips / Total clips: {success_clip}/{total_clips} ({(success_clip / total_clips) * 100 if total_clips > 0 else 0:.2f}%)"
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
    # OUTPUT_DIR = f"./video_anno_{DATASET_NAME}_tmp_openai"
    # OUTPUT_FILE = f"/share/UniHand/sft_data/{DATASET_NAME}_annotations_openai.json"
    OUTPUT_DIR = "/share/HaWoR/cc/openai"
    OUTPUT_FILE = "/share/HaWoR/cc/openai/annotations_openai.json"
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

    pipeline = VideoAnnotationPipeline(
        task_log_file_path='/share/HaWoR/cc/hdf5/epic_kitchens_process_task.json',
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
