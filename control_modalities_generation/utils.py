import colorsys
import json

import cv2
import numpy as np


def get_sample_indices(total_frames, target_frames):
    """Return indices for up to target_frames consecutive frames."""
    if total_frames <= 0:
        return np.array([], dtype=int)

    frames_to_use = min(total_frames, target_frames)

    return np.arange(frames_to_use, dtype=int)


def read_frames(video_path, target_frames):
    """Read sampled frames from a video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}", flush=True)
        return None, None, None

    # Read video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    # Load the frame indices selected by get_sample_indices
    for frame_idx in get_sample_indices(total_frames, target_frames):
        # Move the video reader to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_idx} from {video_path}", flush=True)
            # fallback frame if reading fails
            frame = np.full((height, width, 3), 128, dtype=np.uint8)

        frames.append(frame)

    cap.release()
    return frames, width, height


def create_video_writer(output_path, width, height, fps):
    """Create an MP4 video writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=True,
    )


def get_videos_to_process(
    input_dir,
    output_dir,
    output_suffix,
    required_dir_suffix=None,
):
    """Find input videos that do not already have all required outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() == ".mp4"
    )
    videos = []

    for video_path in video_paths:
        output_path = output_dir / f"{video_path.stem}_{output_suffix}.mp4"

        required_dir_ready = True
        if required_dir_suffix is not None:
            required_dir = output_dir / f"{video_path.stem}_{required_dir_suffix}"
            required_dir_ready = required_dir.exists() and any(required_dir.glob("*.npz"))

        if not output_path.exists() or not required_dir_ready:
            # Skip only when the video and any required sidecar output already exist
            videos.append((video_path, output_path))

    print(f"Found {len(video_paths)} input videos", flush=True)
    print(f"Need to process {len(videos)} videos", flush=True)

    return videos

def load_prompts(prompt_path):
    """Load and flatten prompt classes from a JSON file."""
    with prompt_path.open("r", encoding="utf-8") as file:
        prompt_groups = json.load(file)

    prompts = []

    for classes in prompt_groups.values():
        for class_name in classes:
            if class_name not in prompts:
                prompts.append(class_name)

    return prompts

def get_class_color(class_id):
    """Return a deterministic RGB color for a class ID."""
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)

    return np.array(
        [
            round(red * 255),
            round(green * 255),
            round(blue * 255),
        ],
        dtype=np.uint8,
    )
