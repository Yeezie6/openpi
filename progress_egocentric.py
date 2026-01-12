#!/usr/bin/env python3
"""
处理 Egocentric-10K 数据集的专用脚本
支持视频缩放、裁剪和分块处理
"""
import os
import shutil
import json
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import math
import decord
import time
from tqdm import tqdm

mp.set_start_method('fork', force=True)
os.getcwd
def get_video_info(video_path):
    """Get video information (fps, frame count, duration)"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
        str(video_path)
    ]
    
    try:
        vr = decord.VideoReader(str(video_path))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                fps = eval(stream['r_frame_rate'])
                duration = float(stream.get('duration', 0))
                frame_count = int(stream.get('nb_frames', 0))
                
                if frame_count == 0 and duration > 0:
                    frame_count = int(duration * fps)
                
                return fps, frame_count, duration
        
        raise ValueError("No video stream found")
    except Exception as e:
        print(f"Failed to get video info: {video_path} - {e}")
        return None, None, None

def process_single_video_scale(args):
    """Process single video scaling
    - If scale_factor is provided, scale both width and height by factor.
    - Else, scale short side to target size.
    """
    mp4_file, output_file, target_size, scale_factor, is_intermediate_step = args
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果输出文件已存在且大小合理，跳过
    if output_file.exists() and output_file.stat().st_size > 1000:
        return f"Skipped (exists): {output_file.name}"
    
    # 构建缩放滤镜
    if scale_factor is not None and scale_factor > 0:
        video_filter = (
            f"scale='ceil(iw*{scale_factor}/2)*2':'ceil(ih*{scale_factor}/2)*2'"
        )
    else:
        # 将短边缩放到 target_size
        video_filter = (
            f"scale=w='if(gte(iw,ih), ceil((iw*{target_size})/(ih*2))*2, {target_size})':"
            f"h='if(gte(iw,ih), {target_size}, ceil((ih*{target_size})/(iw*2))*2)'"
        )
    
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(mp4_file),
        "-threads", "2",
        "-vf", video_filter,
        "-c:v", "libx264", "-crf", "20",
        "-c:a", "copy",
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if is_intermediate_step and mp4_file.exists():
            mp4_file.unlink()  # 删除原视频文件（仅中间过程）
        return f"Completed: {output_file.name}"
    except subprocess.TimeoutExpired:
        print(f"[SCALE TIMEOUT] {mp4_file.name}")
        return f"Timeout: {mp4_file.name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"[SCALE ERROR] {mp4_file.name}: {error_msg[:200]}")
        return f"Error: {mp4_file.name} - {error_msg[:100]}"
    except Exception as e:
        print(f"[SCALE EXCEPTION] {mp4_file.name}: {str(e)}")
        return f"Exception: {mp4_file.name} - {str(e)}"

def scale_videos_parallel(source_dir, output_dir, target_size=512, scale_factor=None, max_workers=64, process_id=-1, process_total=1, max_videos=None, is_intermediate_step=False):
    """
    Scale videos to target resolution in parallel
    
    Args:
        source_dir: Source directory containing videos
        output_dir: Output directory for scaled videos
        target_size: Target size for short side
        max_workers: Number of parallel workers
        process_id: Process ID for distributed processing (-1 for single process)
        process_total: Total number of processes for distributed processing
        max_videos: Maximum number of videos to process (None for all)
        is_intermediate_step: Whether this is an intermediate step
    """
    if scale_factor is not None and scale_factor > 0:
        print(f"Starting parallel video scaling by factor={scale_factor}, using {max_workers} workers...")
    else:
        print(f"Starting parallel video scaling to short side={target_size}, using {max_workers} workers...")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return
    
    # Collect all video files to process
    tasks = []
    for mp4_file in source_path.rglob("*.mp4"):
        relative_path = mp4_file.relative_to(source_path)
        output_file = output_path / relative_path
        tasks.append((mp4_file, output_file, target_size, scale_factor, is_intermediate_step))
    
    tasks.sort(key=lambda x: x[0].name)
    
    if process_id >= 0:
        tasks = tasks[process_id::process_total]
        print(f"Process {process_id}/{process_total}: assigned {len(tasks)} videos")
    
    # Limit number of videos if specified
    if max_videos is not None and max_videos > 0:
        tasks = tasks[:max_videos]
        print(f"Limiting to first {max_videos} videos")
    
    print(f"Total video files to process: {len(tasks)}")
    
    if not tasks:
        print("No video files found")
        return
    
    # Process in parallel with progress bar
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Scaling videos", unit="video") as pbar:
            for result in executor.map(process_single_video_scale, tasks):
                completed += 1
                pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                pbar.update(1)

def process_single_video_resize_crop(args):
    """Process single video: resize short side and center crop"""
    mp4_file, output_file, short_side_size, crop_size, is_intermediate_step = args

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 如果输出文件已存在且大小合理，跳过
    if output_file.exists() and output_file.stat().st_size > 1000:
        return f"Skipped (exists): {output_file.name}"
        
    video_filter = (
        f"crop={short_side_size}:{short_side_size}:(iw-{short_side_size})/2:(ih-{short_side_size})/2,"
        f"scale={crop_size}:{crop_size}"
    )
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(mp4_file),
        "-threads", "2",
        "-vf", video_filter,
        "-r", "30",  # 强制输出帧率为30fps
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(output_file)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if is_intermediate_step and mp4_file.exists():
            mp4_file.unlink()  # 删除原视频文件（仅中间过程）
        return f"Completed: {output_file.name}"
    except subprocess.TimeoutExpired:
        print(f"[CROP TIMEOUT] {mp4_file.name}")
        return f"Timeout: {mp4_file.name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"[CROP ERROR] {mp4_file.name}: {error_msg[:200]}")
        return f"Error: {mp4_file.name} - {error_msg[:100]}"
    except Exception as e:
        print(f"[CROP EXCEPTION] {mp4_file.name}: {str(e)}")
        return f"Exception: {mp4_file.name} - {str(e)}"

def process_single_video_crop_rect(args):
    """Process single video: center crop to given width and height (no scaling)"""
    mp4_file, output_file, crop_w, crop_h, is_intermediate_step = args

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and output_file.stat().st_size > 1000:
        return f"Skipped (exists): {output_file.name}"

    # 仅中心裁剪为指定矩形尺寸
    video_filter = f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/2"

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(mp4_file),
        "-threads", "2",
        "-vf", video_filter,
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(output_file)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if is_intermediate_step and mp4_file.exists():
            mp4_file.unlink()  # 删除原视频文件（仅中间过程）
        return f"Completed: {output_file.name}"
    except subprocess.TimeoutExpired:
        print(f"[CROP_RECT TIMEOUT] {mp4_file.name}")
        return f"Timeout: {mp4_file.name}"
    except subprocess.CalledProcessError as e:
        # 获取输入视频尺寸用于错误信息
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(mp4_file)]
            size_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            actual_size = size_result.stdout.strip()
            error_msg = f"input size {actual_size}, required {crop_w}x{crop_h}"
            print(f"[CROP_RECT ERROR] {mp4_file.name}: {error_msg}")
            return f"Error: {mp4_file.name} ({error_msg})"
        except:
            error_msg = e.stderr if e.stderr else str(e)
            print(f"[CROP_RECT ERROR] {mp4_file.name}: {error_msg[:200]}")
            return f"Error: {mp4_file.name} (crop {crop_w}x{crop_h} failed)"
    except Exception as e:
        print(f"[CROP_RECT EXCEPTION] {mp4_file.name}: {str(e)}")
        return f"Exception: {mp4_file.name} - {str(e)}"

def batch_resize_crop_videos(source_dir, output_dir, short_side_size=512, crop_size=256, max_workers=8, process_id=-1, process_total=1, max_videos=None, is_intermediate_step=False):
    """
    批量处理视频：将短边缩放到指定尺寸，然后中心裁切到指定分辨率
    
    Args:
        source_dir: 源视频目录
        output_dir: 输出视频目录
        short_side_size: 短边缩放的目标尺寸
        crop_size: 裁切的目标尺寸（正方形）
        max_workers: 并行工作线程数
        process_id: 进程ID用于分布式处理
        process_total: 总进程数
        max_videos: Maximum number of videos to process (None for all)
        is_intermediate_step: 是否为中间步骤
    """
    print(f"Starting batch video resize and crop...")
    print(f"Short side resize to: {short_side_size}")
    print(f"Center crop to: {crop_size}x{crop_size}")
    print(f"Using {max_workers} workers")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return

    # 收集所有需要处理的视频文件
    tasks = []
    for mp4_file in source_path.rglob("*.mp4"):
        if '/tmp/' in str(mp4_file):
            continue
        relative_path = mp4_file.relative_to(source_path)
        output_file = output_path / relative_path
        tasks.append((mp4_file, output_file, short_side_size, crop_size, is_intermediate_step))

    tasks.sort(key=lambda x: x[0].name)
    
    if process_id >= 0:
        tasks = tasks[process_id::process_total]
        print(f"Process {process_id}/{process_total}: assigned {len(tasks)} videos")
        
    # Limit number of videos if specified
    if max_videos is not None and max_videos > 0:
        tasks = tasks[:max_videos]
        print(f"Limiting to first {max_videos} videos")
        
    print(f"Total video files to process: {len(tasks)}")
    
    if not tasks:
        print("No video files found to process")
        return

    # 并行处理 with progress bar
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Resize & Crop videos", unit="video") as pbar:
            for result in executor.map(process_single_video_resize_crop, tasks):
                completed += 1
                pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                pbar.update(1)

    print(f"\n=== Processing completed ===")
    print(f"Total processed: {completed}/{len(tasks)}")

def process_single_video_chunk_frames(args):
    """Process single video: chunk by frame count"""
    mp4_file, output_dir, frames_per_chunk, delete_original, is_intermediate_step = args
    
    try:
        # 获取视频信息
        vr = decord.VideoReader(str(mp4_file))
        fps = vr.get_avg_fps()
        frame_count = len(vr)
        
        base_name = mp4_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_chunks = math.ceil(frame_count / frames_per_chunk)
        
        # 检查是否所有chunk都已存在
        all_exist = True
        for chunk_id in range(num_chunks):
            output_chunk = output_dir / f"{base_name}_{chunk_id:04d}.mp4"
            if not output_chunk.exists() or output_chunk.stat().st_size == 0:
                all_exist = False
                break
        
        if all_exist:
            if delete_original and mp4_file.exists():
                mp4_file.unlink()
            return f"Skipped (all chunks exist): {mp4_file.name}"
        
        # 逐个生成chunk，使用精确的帧提取
        for chunk_id in range(num_chunks):
            output_chunk = output_dir / f"{base_name}_{chunk_id:04d}.mp4"
            
            # 如果chunk已存在且大小合理，跳过
            if output_chunk.exists() and output_chunk.stat().st_size > 1000:
                continue
            
            start_frame = chunk_id * frames_per_chunk
            end_frame = min(start_frame + frames_per_chunk, frame_count)
            actual_frames = end_frame - start_frame
            
            # 使用 -ss 和 -frames:v 精确提取帧
            start_time = start_frame / fps
            
            cmd = [
                "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                "-ss", str(start_time),  # 从指定时间开始
                "-i", str(mp4_file),
                "-frames:v", str(actual_frames),  # 精确提取帧数
                "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_chunk)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        if is_intermediate_step and delete_original and mp4_file.exists():
            mp4_file.unlink()  # 删除原视频文件（仅中间过程）
        
        return f"Completed: {mp4_file.name} -> {num_chunks} chunks"
        
    except subprocess.TimeoutExpired:
        print(f"[CHUNK TIMEOUT] {mp4_file.name}")
        return f"Timeout: {mp4_file.name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"[CHUNK ERROR] {mp4_file.name}: {error_msg[:200]}")
        return f"Error: {mp4_file.name} - {error_msg[:100]}"
    except Exception as e:
        print(f"[CHUNK EXCEPTION] {mp4_file.name}: {str(e)}")
        return f"Exception: {mp4_file.name} - {str(e)}"

def chunk_videos_parallel(source_dir, output_dir, frames_per_chunk=300, max_workers=8, 
                         delete_original=False, process_id=-1, process_total=1, max_videos=None, is_intermediate_step=False):
    """
    Chunk videos in parallel by frame count
    
    Args:
        source_dir: Source directory containing videos
        output_dir: Output directory for chunked videos
        frames_per_chunk: Number of frames per chunk
        max_workers: Number of parallel workers
        delete_original: Whether to delete original videos after processing
        process_id: Process ID for distributed processing
        process_total: Total number of processes
        max_videos: Maximum number of videos to process (None for all)
        is_intermediate_step: Whether this is an intermediate step
    """
    print(f"Starting parallel video chunking...")
    print(f"Frames per chunk: {frames_per_chunk}")
    print(f"Using {max_workers} workers")
    print(f"Delete original: {delete_original}")

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return
    
    # Collect all video files to process
    tasks = []
    for mp4_file in source_path.rglob("*.mp4"):
        relative_path = mp4_file.relative_to(source_path)
        output_subdir = output_path / relative_path.parent
        tasks.append((mp4_file, output_subdir, frames_per_chunk, delete_original, is_intermediate_step))
    
    tasks.sort(key=lambda x: x[0].name)
    
    if process_id >= 0:
        tasks = tasks[process_id::process_total]
        print(f"Process {process_id}/{process_total}: assigned {len(tasks)} videos")
        
    # Limit number of videos if specified
    if max_videos is not None and max_videos > 0:
        tasks = tasks[:max_videos]
        print(f"Limiting to first {max_videos} videos")
        
    print(f"Total video files to process: {len(tasks)}")
    
    if not tasks:
        print("No video files found to process")
        return
    
    # Process all tasks with progress bar
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Chunking videos", unit="video") as pbar:
            for result in executor.map(process_single_video_chunk_frames, tasks):
                completed += 1
                pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                pbar.update(1)
    
    print(f"\n=== Chunking completed ===")
    print(f"Total processed: {completed}/{len(tasks)}")

def process_single_video_enhance(args):
    """Process single video: enhance clarity with denoising and sharpening"""
    mp4_file, output_file, enhance_params, is_intermediate_step = args
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果输出文件已存在且大小合理，跳过
    if output_file.exists() and output_file.stat().st_size > 1000:
        return f"Skipped (exists): {output_file.name}"
    
    # 构建视频滤镜：去噪 + 锐化 + 对比度增强
    filters = []
    
    # 1. 去噪（可选）
    if enhance_params.get('denoise', True):
        denoise_strength = enhance_params.get('denoise_strength', 'medium')
        if denoise_strength == 'light':
            filters.append("hqdn3d=2:1.5:3:2.25")
        elif denoise_strength == 'medium':
            filters.append("hqdn3d=4:3:6:4.5")
        elif denoise_strength == 'strong':
            filters.append("hqdn3d=8:6:12:9")
    
    # 2. 锐化
    if enhance_params.get('sharpen', True):
        sharpen_strength = enhance_params.get('sharpen_strength', 'medium')
        if sharpen_strength == 'light':
            filters.append("unsharp=5:5:0.5:5:5:0.0")
        elif sharpen_strength == 'medium':
            filters.append("unsharp=5:5:1.0:5:5:0.0")
        elif sharpen_strength == 'strong':
            filters.append("unsharp=5:5:1.5:5:5:0.0")
    
    # 3. 对比度和饱和度增强（可选）
    if enhance_params.get('contrast', False):
        contrast_value = enhance_params.get('contrast_value', 1.1)
        brightness = enhance_params.get('brightness', 0.0)
        saturation = enhance_params.get('saturation', 1.0)
        filters.append(f"eq=contrast={contrast_value}:brightness={brightness}:saturation={saturation}")
    
    video_filter = ",".join(filters) if filters else "copy"
    
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(mp4_file),
        "-threads", "2",
        "-vf", video_filter,
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if is_intermediate_step and mp4_file.exists():
            mp4_file.unlink()  # 删除原视频文件（仅中间过程）
        return f"Completed: {output_file.name}"
    except subprocess.TimeoutExpired:
        print(f"[ENHANCE TIMEOUT] {mp4_file.name}")
        return f"Timeout: {mp4_file.name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"[ENHANCE ERROR] {mp4_file.name}: {error_msg[:200]}")
        return f"Error: {mp4_file.name} - {error_msg[:100]}"
    except Exception as e:
        print(f"[ENHANCE EXCEPTION] {mp4_file.name}: {str(e)}")
        return f"Exception: {mp4_file.name} - {str(e)}"

def enhance_videos_parallel(source_dir, output_dir, max_workers=8, enhance_params=None,
                           process_id=-1, process_total=1, max_videos=None, is_intermediate_step=False):
    """
    Enhance video clarity with denoising and sharpening in parallel
    
    Args:
        source_dir: Source directory containing videos
        output_dir: Output directory for enhanced videos
        max_workers: Number of parallel workers
        enhance_params: Dictionary of enhancement parameters
        process_id: Process ID for distributed processing
        process_total: Total number of processes
        max_videos: Maximum number of videos to process (None for all)
        is_intermediate_step: Whether this is an intermediate step
    """
    if enhance_params is None:
        enhance_params = {
            'denoise': True,
            'denoise_strength': 'medium',  # light, medium, strong
            'sharpen': True,
            'sharpen_strength': 'medium',  # light, medium, strong
            'contrast': False,
            'contrast_value': 1.1,
            'brightness': 0.0,
            'saturation': 1.0
        }
    
    print(f"Starting video enhancement, using {max_workers} workers...")
    print(f"Enhancement parameters: denoise={enhance_params['denoise']}, sharpen={enhance_params['sharpen']}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return
    
    # Collect all video files to process
    tasks = []
    for mp4_file in source_path.rglob("*.mp4"):
        relative_path = mp4_file.relative_to(source_path)
        output_file = output_path / relative_path
        tasks.append((mp4_file, output_file, enhance_params, is_intermediate_step))
    
    tasks.sort(key=lambda x: x[0].name)
    
    if process_id >= 0:
        tasks = tasks[process_id::process_total]
        print(f"Process {process_id}/{process_total}: assigned {len(tasks)} videos")
    
    if max_videos is not None and max_videos > 0:
        tasks = tasks[:max_videos]
        print(f"Limiting to first {max_videos} videos")
    
    print(f"Total videos to enhance: {len(tasks)}")
    
    if not tasks:
        print("No videos found to enhance")
        return
    
    # Process all tasks with progress bar
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Enhancing videos", unit="video") as pbar:
            for result in executor.map(process_single_video_enhance, tasks):
                completed += 1
                pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                pbar.update(1)
    
    print(f"\n=== Enhancement completed ===")
    print(f"Total processed: {completed}/{len(tasks)}")

def process_single_video_fisheye(args):
    """Process single video: fisheye to rectilinear projection"""
    input_video, output_video, fisheye_params, is_intermediate_step = args
    
    try:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果输出文件已存在且大小合理，跳过
        if output_video.exists() and output_video.stat().st_size > 1000:
            return f"Skipped (exists): {output_video.name}"
        
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", str(input_video),
            "-vf", f"v360=fisheye:rectilinear:ih_fov={fisheye_params['ih_fov']}:iv_fov={fisheye_params['iv_fov']}:h_fov={fisheye_params['h_fov']}:v_fov={fisheye_params['v_fov']}:w={fisheye_params['w']}:h={fisheye_params['h']}:interp={fisheye_params['interp']}",
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-an",
            "-threads", "4",  # 增加线程数以加快处理
            str(output_video)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)  # 增加到30分钟超时
        if is_intermediate_step and input_video.exists():
            input_video.unlink()  # 删除原视频文件（仅中间过程）
        return f"Completed fisheye conversion: {output_video.name}"
        
    except subprocess.TimeoutExpired:
        print(f"[FISHEYE TIMEOUT] {input_video.name}")
        return f"Timeout: {input_video.name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"[FISHEYE ERROR] {input_video.name}: {error_msg[:200]}")
        return f"Error: {input_video.name} - {error_msg[:100]}"
    except Exception as e:
        print(f"[FISHEYE EXCEPTION] {input_video.name}: {str(e)}")
        return f"Exception: {input_video.name} - {str(e)}"

def process_fisheye_videos(source_dir, output_dir, max_workers=32, fisheye_params=None, 
                          process_id=-1, process_total=1, max_videos=None, is_intermediate_step=False):
    """
    Process fisheye to rectilinear projection for videos in parallel
    
    Args:
        source_dir: Source directory containing fisheye videos
        output_dir: Output directory for converted videos
        max_workers: Number of parallel workers
        fisheye_params: Dictionary of fisheye conversion parameters
        process_id: Process ID for distributed processing
        process_total: Total number of processes
        max_videos: Maximum number of videos to process (None for all)
        is_intermediate_step: Whether this is an intermediate step
    """
    if fisheye_params is None:
        fisheye_params = {
            'ih_fov': 110,      # Input horizontal field of view
            'iv_fov': 110,      # Input vertical field of view
            'h_fov': 90,        # Output horizontal field of view
            'v_fov': 90,        # Output vertical field of view
            'w': 1408,          # Output width
            'h': 1408,          # Output height
            'interp': 'lanczos' # Interpolation method
        }
    
    print(f"Starting fisheye conversion, using {max_workers} workers...")
    print(f"Fisheye parameters: {fisheye_params}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return
    
    # Collect all video files to process
    tasks = []
    for mp4_file in source_path.rglob("*.mp4"):
        relative_path = mp4_file.relative_to(source_path)
        output_file = output_path / relative_path
        tasks.append((mp4_file, output_file, fisheye_params, is_intermediate_step))
    
    tasks.sort(key=lambda x: x[0].name)
    
    if process_id >= 0:
        tasks = tasks[process_id::process_total]
        print(f"Process {process_id}/{process_total}: assigned {len(tasks)} videos")
    
    # Limit number of videos if specified
    if max_videos is not None and max_videos > 0:
        tasks = tasks[:max_videos]
        print(f"Limiting to first {max_videos} videos")
    
    print(f"Total fisheye videos to process: {len(tasks)}")
    
    if not tasks:
        print("No fisheye videos found to process")
        return
    
    # Process all tasks with progress bar
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Fisheye conversion", unit="video") as pbar:
            for result in executor.map(process_single_video_fisheye, tasks):
                completed += 1
                pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                pbar.update(1)
    
    print(f"\n=== Fisheye conversion completed ===")
    print(f"Total processed: {completed}/{len(tasks)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="处理 Egocentric-10K 视频数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理模式:
  scale:   缩放视频短边到指定尺寸
  crop:    先缩放后中心裁剪到正方形
  chunk:   按帧数分块视频
  fisheye: 鱼眼镜头转换为矩形投影
  enhance: 视频增强（去噪+锐化+对比度）
  
可以指定多个模式，按顺序执行（用逗号分隔），例如: --mode enhance,fisheye,scale,chunk
  
示例:
  # 单个模式：视频增强（去噪+锐化）
  python process_egocentric.py --mode enhance --source /disk0/videos --output /disk0/enhanced --workers 16
  
  # 视频增强（强力锐化+对比度增强）
  python process_egocentric.py --mode enhance --source /disk0/videos --output /disk0/enhanced --sharpen-strength strong --enhance-contrast --contrast-value 1.15
  
  # 单个模式：缩放视频短边到512
  python process_egocentric.py --mode scale --source /disk0/videos --output /disk0/output --target-size 512 --workers 32
  
  # 单个模式：中心裁剪到256x256
  python process_egocentric.py --mode crop --source /disk0/videos --output /disk0/output --short-side 512 --crop-size 256 --workers 16
  
  # 单个模式：分块视频（每300帧）
  python process_egocentric.py --mode chunk --source /disk0/videos --output /disk0/output --frames-per-chunk 300 --workers 8 --delete-original
  
  # 单个模式：鱼眼镜头转换
  python process_egocentric.py --mode fisheye --source /disk0/Egocentric-10K/factory_002 --output /share/HaWoR/example/render --workers 24 --ih-fov 110 --h-fov 90 --output-width 512 --output-height 512
  
  # 流水线模式：鱼眼转换 -> 缩放 -> 分块
  python process_egocentric.py --mode fisheye,scale,chunk --source /disk0/fisheye_videos --output /disk0/final_output --target-size 512 --frames-per-chunk 300 --workers 16
  
  # 流水线模式：缩放 -> 裁剪
  python process_egocentric.py --mode scale,crop --source /disk0/videos --output /disk0/cropped --target-size 512 --crop-size 256 --workers 32
  
  # 保留中间结果：先chunk后crop（保留910×512的分块视频）
  python process_egocentric.py --mode fisheye,scale,chunk,crop --source example/exp/raw --output example/exp/test --target-size 512 --crop-size 256 --frames-per-chunk 300 --keep-intermediate --workers 16
  
  # 仅保留chunk后的结果（删除fisheye和scale的中间文件，保留chunk后的910×512视频和最终256×256视频）
  python process_egocentric.py --mode fisheye,scale,chunk,crop --source example/exp/raw --output example/exp/test --ih-fov 105 --iv-fov 95 --h-fov 95 --v-fov 90 --output-width 910 --output-height 512 --target-size 512 --crop-size 256 --frames-per-chunk 300 --keep-chunked --workers 16
        """
    )
    
    parser.add_argument('--mode', required=True, type=str,
                       help='处理模式，可以是单个模式或逗号分隔的多个模式 (fisheye,scale,crop,chunk)')
    parser.add_argument('--source', required=True,
                       help='源视频目录')
    parser.add_argument('--output', required=True,
                       help='输出目录')
    
    # Scale mode arguments
    parser.add_argument('--target-size', type=int, default=512,
                       help='缩放目标尺寸（短边）（默认: 512）')
    parser.add_argument('--scale-factor', type=float, default=None,
                       help='按比例缩放因子（宽高同时按此因子缩放，例如 1.46）')
    
    # Crop mode arguments
    parser.add_argument('--short-side', type=int, default=512,
                       help='裁剪前短边尺寸（默认: 512）')
    parser.add_argument('--crop-size', type=int, default=256,
                       help='裁剪后正方形尺寸（默认: 256）')
    parser.add_argument('--crop-width', type=int, default=None,
                       help='矩形中心裁剪宽度（与 crop-height 一起使用）')
    parser.add_argument('--crop-height', type=int, default=None,
                       help='矩形中心裁剪高度（与 crop-width 一起使用）')
    
    # Chunk mode arguments
    parser.add_argument('--frames-per-chunk', type=int, default=300,
                       help='每个分块的帧数（默认: 300）')
    parser.add_argument('--delete-original', action='store_true',
                       help='分块后删除原始视频')
    
    # Fisheye mode arguments
    parser.add_argument('--ih-fov', type=int, default=95,
                       help='输入水平视场角')
    parser.add_argument('--iv-fov', type=int, default=95,
                       help='输入垂直视场角')
    parser.add_argument('--h-fov', type=int, default=70,
                       help='输出水平视场角')
    parser.add_argument('--v-fov', type=int, default=70,
                       help='输出垂直视场角')
    parser.add_argument('--output-width', type=int, default=1024,
                       help='输出视频宽度（默认: 512）')
    parser.add_argument('--output-height', type=int, default=1024,
                       help='输出视频高度（默认: 512）')
    parser.add_argument('--interp', type=str, default='lanczos',
                       choices=['lanczos', 'cubic', 'linear'],
                       help='插值方法（默认: cubic）')
    
    # Enhance mode arguments
    parser.add_argument('--denoise', action='store_true', default=True,
                       help='启用去噪（默认: True）')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false',
                       help='禁用去噪')
    parser.add_argument('--denoise-strength', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='去噪强度（默认: medium）')
    parser.add_argument('--sharpen', action='store_true', default=True,
                       help='启用锐化（默认: True）')
    parser.add_argument('--no-sharpen', dest='sharpen', action='store_false',
                       help='禁用锐化')
    parser.add_argument('--sharpen-strength', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='锐化强度（默认: medium）')
    parser.add_argument('--enhance-contrast', action='store_true',
                       help='启用对比度增强')
    parser.add_argument('--contrast-value', type=float, default=1.1,
                       help='对比度值（默认: 1.1）')
    parser.add_argument('--brightness', type=float, default=0.0,
                       help='亮度调整（默认: 0.0）')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='饱和度（默认: 1.0）')
    
    # Common arguments
    parser.add_argument('--workers', type=int, default=8,
                       help='并行工作线程数（默认: 8）')
    parser.add_argument('--process-id', type=int, default=-1,
                       help='分布式处理的进程ID（-1表示单进程）')
    parser.add_argument('--process-total', type=int, default=1,
                       help='分布式处理的总进程数')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='限制处理的视频数量（默认: 处理所有视频）')
    parser.add_argument('--keep-intermediate', action='store_true',
                       help='保留中间临时文件（不自动清理 _temp_step_* 目录）')
    parser.add_argument('--keep-chunked', action='store_true',
                       help='仅保留chunk步骤之后的中间结果（删除chunk之前的临时文件）')
    parser.add_argument('--keep-penultimate', action='store_true',
                       help='仅保留倒数第二步（最后一步之前）的中间结果')
    
    args = parser.parse_args()
    
    # 检测CPU核心数并给出建议
    cpu_count = mp.cpu_count()
    print(f"检测到 CPU 核心数: {cpu_count}")
    if args.workers > cpu_count:
        print(f"警告: 指定的工作线程数 ({args.workers}) 超过 CPU 核心数 ({cpu_count})")
    
    # 解析处理模式（支持逗号分隔的多个模式）
    modes = [m.strip() for m in args.mode.split(',')]
    
    # 验证所有模式都是有效的
    valid_modes = ['scale', 'crop', 'crop_rect', 'chunk', 'fisheye', 'enhance']
    for mode in modes:
        if mode not in valid_modes:
            print(f"错误: 无效的处理模式 '{mode}'")
            print(f"有效模式: {', '.join(valid_modes)}")
            return
    
    print(f"\n{'='*60}")
    print(f"处理流水线: {' -> '.join(modes)}")
    print(f"{'='*60}\n")
    
    # 设置初始输入目录
    current_source = args.source
    
    # 按顺序执行每个模式
    for idx, mode in enumerate(modes):
        is_last_step = (idx == len(modes) - 1)
        # 根据要求：第一步和最后一步不删除原视频文件，中间步则在处理完后立即删除
        is_first_step = (idx == 0)
        should_delete_input = not is_first_step and not is_last_step

        # 为中间步骤创建临时输出目录
        if is_last_step:
            current_output = args.output
        else:
            current_output = str(Path(args.output) / f"_temp_step_{idx}_{mode}")

        print(f"\n{'='*60}")
        print(f"步骤 {idx + 1}/{len(modes)}: {mode.upper()}")
        print(f"输入: {current_source}")
        print(f"输出: {current_output}")
        print(f"{'='*60}\n")
        
        if mode == 'scale':
            scale_videos_parallel(
                source_dir=current_source,
                output_dir=current_output,
                target_size=args.target_size,
                scale_factor=args.scale_factor,
                max_workers=args.workers,
                process_id=args.process_id,
                process_total=args.process_total,
                max_videos=args.max_videos,
                is_intermediate_step=should_delete_input
            )
        
        elif mode == 'crop':
            batch_resize_crop_videos(
                source_dir=current_source,
                output_dir=current_output,
                short_side_size=args.short_side,
                crop_size=args.crop_size,
                max_workers=args.workers,
                process_id=args.process_id,
                process_total=args.process_total,
                max_videos=args.max_videos,
                is_intermediate_step=should_delete_input
            )
        elif mode == 'crop_rect':
            # 中心矩形裁剪（不缩放），需要 crop_width 与 crop_height
            if args.crop_width is None or args.crop_height is None:
                print("错误: 使用 crop_rect 模式需要提供 --crop-width 与 --crop-height")
                return
            # 收集任务
            source_path = Path(current_source)
            output_path = Path(current_output)
            tasks = []
            for mp4_file in source_path.rglob("*.mp4"):
                if '/tmp/' in str(mp4_file):
                    continue
                relative_path = mp4_file.relative_to(source_path)
                output_file = output_path / relative_path
                tasks.append((mp4_file, output_file, args.crop_width, args.crop_height, should_delete_input))

            tasks.sort(key=lambda x: x[0].name)

            if args.process_id >= 0:
                tasks = tasks[args.process_id::args.process_total]
                print(f"Process {args.process_id}/{args.process_total}: assigned {len(tasks)} videos")

            print(f"Total videos to rect-crop: {len(tasks)}")
            if not tasks:
                print("No videos found to rect-crop")
            else:
                completed = 0
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    with tqdm(total=len(tasks), desc="Rect Crop videos", unit="video") as pbar:
                        for result in executor.map(process_single_video_crop_rect, tasks):
                            completed += 1
                            pbar.set_postfix_str(result.split(':')[0] if ':' in result else result)
                            pbar.update(1)
        
        elif mode == 'chunk':
            chunk_videos_parallel(
                source_dir=current_source,
                output_dir=current_output,
                frames_per_chunk=args.frames_per_chunk,
                max_workers=args.workers,
                delete_original=args.delete_original or should_delete_input,
                process_id=args.process_id,
                process_total=args.process_total,
                max_videos=args.max_videos,
                is_intermediate_step=should_delete_input
            )
        
        elif mode == 'fisheye':
            fisheye_params = {
                'ih_fov': args.ih_fov,
                'iv_fov': args.iv_fov,
                'h_fov': args.h_fov,
                'v_fov': args.v_fov,
                'w': args.output_width,
                'h': args.output_height,
                'interp': args.interp
            }
            process_fisheye_videos(
                source_dir=current_source,
                output_dir=current_output,
                max_workers=args.workers,
                fisheye_params=fisheye_params,
                process_id=args.process_id,
                process_total=args.process_total,
                max_videos=args.max_videos,
                is_intermediate_step=should_delete_input
            )
        
        elif mode == 'enhance':
            enhance_params = {
                'denoise': args.denoise,
                'denoise_strength': args.denoise_strength,
                'sharpen': args.sharpen,
                'sharpen_strength': args.sharpen_strength,
                'contrast': args.enhance_contrast,
                'contrast_value': args.contrast_value,
                'brightness': args.brightness,
                'saturation': args.saturation
            }
            enhance_videos_parallel(
                source_dir=current_source,
                output_dir=current_output,
                max_workers=args.workers,
                enhance_params=enhance_params,
                process_id=args.process_id,
                process_total=args.process_total,
                max_videos=args.max_videos,
                is_intermediate_step=should_delete_input
            )
        
        # 更新下一步的输入目录
        current_source = current_output
        
        print(f"\n步骤 {idx + 1} 完成!")
    
    # 清理中间临时文件夹
    if len(modes) > 1:
        if args.keep_penultimate:
            # 只保留倒数第二步的结果
            penultimate_idx = len(modes) - 2
            print(f"\n{'='*60}")
            print("保留倒数第二步的中间结果，清理其他临时文件 (--keep-penultimate)")
            print(f"{'='*60}\n")
            
            # 删除倒数第二步之前的所有临时文件
            for idx in range(penultimate_idx):
                temp_dir = Path(args.output) / f"_temp_step_{idx}_{modes[idx]}"
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"✓ 已删除: {temp_dir}")
                    except Exception as e:
                        print(f"✗ 删除失败 {temp_dir}: {e}")
            
            # 保留倒数第二步
            penultimate_dir = Path(args.output) / f"_temp_step_{penultimate_idx}_{modes[penultimate_idx]}"
            if penultimate_dir.exists():
                print(f"📁 保留: {penultimate_dir}")
        
        elif args.keep_chunked:
            # 找到chunk步骤的索引
            chunk_idx = None
            for idx, mode in enumerate(modes):
                if mode == 'chunk':
                    chunk_idx = idx
                    break
            
            if chunk_idx is not None:
                print(f"\n{'='*60}")
                print("保留chunk之后的中间结果，清理chunk之前的临时文件 (--keep-chunked)")
                print(f"{'='*60}\n")
                
                # 删除chunk之前的所有临时文件
                for idx in range(chunk_idx):
                    temp_dir = Path(args.output) / f"_temp_step_{idx}_{modes[idx]}"
                    if temp_dir.exists():
                        try:
                            shutil.rmtree(temp_dir)
                            print(f"✓ 已删除: {temp_dir}")
                        except Exception as e:
                            print(f"✗ 删除失败 {temp_dir}: {e}")
                
                # 保留chunk及之后的临时文件
                for idx in range(chunk_idx, len(modes) - 1):
                    temp_dir = Path(args.output) / f"_temp_step_{idx}_{modes[idx]}"
                    if temp_dir.exists():
                        print(f"📁 保留: {temp_dir}")
            else:
                print(f"\n⚠️  警告: --keep-chunked 需要在处理模式中包含 'chunk' 步骤")
        
        elif not args.keep_intermediate:
            print(f"\n{'='*60}")
            print("清理中间临时文件...")
            print(f"{'='*60}\n")
            
            for idx in range(len(modes) - 1):
                temp_dir = Path(args.output) / f"_temp_step_{idx}_{modes[idx]}"
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"✓ 已删除: {temp_dir}")
                    except Exception as e:
                        print(f"✗ 删除失败 {temp_dir}: {e}")
        
        elif args.keep_intermediate:
            print(f"\n{'='*60}")
            print("保留中间临时文件 (--keep-intermediate)")
            print(f"{'='*60}\n")
            for idx in range(len(modes) - 1):
                temp_dir = Path(args.output) / f"_temp_step_{idx}_{modes[idx]}"
                if temp_dir.exists():
                    print(f"📁 保留: {temp_dir}")
    
    print(f"\n{'='*60}")
    print(f"✨ 所有处理步骤完成！")
    print(f"最终输出目录: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

"""
python process_egocentric.py \
  --mode fisheye,scale,chunk,crop \
  --source example/exp/raw \
  --output example/exp/test33 \
  --ih-fov 105 --iv-fov 95 \
  --h-fov 95 --v-fov 90 \
  --output-width 910 --output-height 512 \
  --target-size 512 \
  --crop-size 256 \
  --frames-per-chunk 300 \
  --keep-chunked \
  --workers 16 
"""