import os
import math
import glob
import shutil
import subprocess


def get_duration(video_path):
    """
    Return video duration in seconds using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def clean_segments(out_dir):
    """
    Remove old segments and recreate output folder.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)


def count_segments(out_dir):
    return len(glob.glob(os.path.join(out_dir, "*.mp4")))


def segment_fast(video_path, out_dir, step=5):
    """
    Fast segmentation using stream copy.
    Falls back to re-encode if copy mode fails.
    """
    os.makedirs(out_dir, exist_ok=True)

    duration = get_duration(video_path)
    total_segments = math.ceil(duration / step)

    print(f"FAST SEGMENT MODE → {total_segments} segments", flush=True)

    output_pattern = os.path.join(out_dir, "seg_%04d.mp4")

    fast_cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(step),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        output_pattern
    ]

    result = subprocess.run(fast_cmd, capture_output=True, text=True)

    if result.returncode != 0 or count_segments(out_dir) == 0:
        print("Fast copy segmentation failed. Falling back to re-encode mode...", flush=True)

        # clear possibly broken outputs
        for f in glob.glob(os.path.join(out_dir, "*.mp4")):
            try:
                os.remove(f)
            except Exception:
                pass

        slow_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", "0",
            "-f", "segment",
            "-segment_time", str(step),
            "-reset_timestamps", "1",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_pattern
        ]

        result = subprocess.run(slow_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg segmentation failed:\n{result.stderr}")

    print(f"Segmentation complete → {count_segments(out_dir)} files created", flush=True)
    return duration


def segment_full(video_path, out_dir, step=5):
    """
    Reliable segmentation using re-encoding.
    Better for accurate cuts and stable downstream processing.
    """
    os.makedirs(out_dir, exist_ok=True)

    duration = get_duration(video_path)
    total_segments = math.ceil(duration / step)

    print(f"FULL SEGMENT MODE → {total_segments} segments", flush=True)

    output_pattern = os.path.join(out_dir, "seg_%04d.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(step),
        "-reset_timestamps", "1",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        output_pattern
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg full segmentation failed:\n{result.stderr}")

    print(f"Segmentation complete → {count_segments(out_dir)} files created", flush=True)
    return duration


def segment(video_path, out_dir, seg_len=5, fast=True):
    """
    Main pipeline entry.
    Default: fast mode.
    
    Example:
        segment(video_path, out_dir)
        segment(video_path, out_dir, seg_len=8, fast=True)
        segment(video_path, out_dir, seg_len=5, fast=False)
    """
    clean_segments(out_dir)

    if fast:
        return segment_fast(video_path, out_dir, step=seg_len)
    return segment_full(video_path, out_dir, step=seg_len)