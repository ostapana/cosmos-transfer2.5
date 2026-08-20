# This script generates monocular depth maps for the videos specified in dataset folder
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from video_utils import create_video_writer, get_videos_to_process, read_frames


INPUT_VIDEO_DIR = Path("data/video_dataset")
OUTPUT_DEPTH_DIR = Path("data/depth_controls")

MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"

TARGET_FPS = 10
TARGET_FRAMES = 100


def frame_to_pil(frame):
    """Convert an OpenCV BGR frame to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def normalize_depth(result):
    """Convert model depth output to an 8-bit grayscale image."""
    depth = np.array(result["depth"])
    return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def write_depth_video(results, output_path, width, height):
    """Save depth estimation results as a video."""
    writer = create_video_writer(output_path, width, height, TARGET_FPS)

    for result in results:
        depth = normalize_depth(result)

        # Resize the depth map if the model output resolution differs from the video frame
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)

        writer.write(cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR))

    writer.release()


def process_video(estimator, video_path, output_path):
    """Estimate depth for one video and save the result."""
    frames, width, height = read_frames(video_path, TARGET_FRAMES)

    if frames is None:
        return

    print(f"Estimating depth: {video_path.name}", flush=True)

    results = estimator([frame_to_pil(frame) for frame in frames])
    write_depth_video(results, output_path, width, height)

    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    # High-level wrapper for loading and running the depth model
    estimator = pipeline("depth-estimation", model=MODEL_NAME, device=0)
    videos = get_videos_to_process(
        INPUT_VIDEO_DIR,
        OUTPUT_DEPTH_DIR,
        "depth",
    )

    for video_path, output_path in videos:
        process_video(estimator, video_path, output_path)

    print("Done", flush=True)
    