from pathlib import Path
import multiprocessing as mp
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import torch

INPUT_VIDEO_DIR = Path("cosmos_transfer2/dataset/videos_low_res")
OUTPUT_DEPTH_DIR = Path("cosmos_transfer2/dataset/depth_low_res")

MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"

VIDEO_EXTS = {".mp4"}

TARGET_FPS = 10
TARGET_FRAMES = 101

REQUESTED_GPUS = 8
BATCH_SIZE = 8


def frame_to_pil_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def depth_result_to_u8(result):
    depth_np = np.array(result["depth"])

    depth_u8 = cv2.normalize(
        depth_np,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype("uint8")

    return depth_u8


def sample_frame_indices(total_frames, target_frames):
    if total_frames <= 0:
        raise ValueError("Video has no frames")

    if total_frames >= target_frames:
        return np.linspace(0, total_frames - 1, target_frames).round().astype(int)

    indices = list(range(total_frames))
    indices += [total_frames - 1] * (target_frames - total_frames)
    return np.array(indices, dtype=int)


def read_sampled_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}", flush=True)
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices = sample_frame_indices(total_frames, TARGET_FRAMES)

    frames = []

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok:
            print(
                f"Could not read frame {frame_idx} from {video_path}, using gray fallback",
                flush=True,
            )
            frame = np.full((height, width, 3), 128, dtype=np.uint8)

        frames.append(frame)

    cap.release()

    return frames, width, height


def process_video(estimator, video_path, output_path, gpu_id):
    data = read_sampled_frames(video_path)

    if data[0] is None:
        return

    frames, width, height = data

    pil_frames = [frame_to_pil_bgr(frame) for frame in frames]

    print(
        f"[GPU {gpu_id}] Estimating depth: {video_path.name} with batch size {BATCH_SIZE}",
        flush=True,
    )

    results = estimator(
        pil_frames,
        batch_size=BATCH_SIZE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (width, height),
        isColor=True,
    )

    for result in results:
        depth_u8 = depth_result_to_u8(result)

        if depth_u8.shape[:2] != (height, width):
            depth_u8 = cv2.resize(
                depth_u8,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )

        depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
        writer.write(depth_bgr)

    writer.release()

    print(
        f"[GPU {gpu_id}] Saved {output_path} ({TARGET_FRAMES} frames, {TARGET_FPS} fps, {width}x{height})",
        flush=True,
    )


def worker(gpu_id, videos_to_process):
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Starting worker", flush=True)
    print(f"[GPU {gpu_id}] Device name: {torch.cuda.get_device_name(gpu_id)}", flush=True)
    print(f"[GPU {gpu_id}] Videos assigned: {len(videos_to_process)}", flush=True)

    estimator = hf_pipeline(
        "depth-estimation",
        model=MODEL_NAME,
        device=gpu_id,
    )

    for video_path, output_path in videos_to_process:
        if output_path.exists():
            print(f"[GPU {gpu_id}] Skipping existing: {output_path}", flush=True)
            continue

        process_video(estimator, video_path, output_path, gpu_id)

    print(f"[GPU {gpu_id}] Worker done", flush=True)


def main():
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    available_gpus = torch.cuda.device_count()
    gpus_to_use = min(REQUESTED_GPUS, available_gpus)

    if gpus_to_use <= 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"Available GPUs: {available_gpus}", flush=True)
    print(f"Requested GPUs: {REQUESTED_GPUS}", flush=True)
    print(f"GPUs that will be used: {gpus_to_use}", flush=True)
    print(f"Pipeline batch size per GPU: {BATCH_SIZE}", flush=True)

    for gpu_id in range(gpus_to_use):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)

    OUTPUT_DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in INPUT_VIDEO_DIR.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )

    if not video_paths:
        raise RuntimeError(f"No videos found in {INPUT_VIDEO_DIR}")

    videos_to_process = []

    for video_path in video_paths:
        output_path = OUTPUT_DEPTH_DIR / f"{video_path.stem}_depth.mp4"

        if output_path.exists():
            continue

        videos_to_process.append((video_path, output_path))

    print(f"Found {len(video_paths)} input videos", flush=True)
    print(f"Need to process {len(videos_to_process)} videos", flush=True)

    if not videos_to_process:
        print("All depth videos already exist. Nothing to do.", flush=True)
        return

    assignments = [[] for _ in range(gpus_to_use)]

    for idx, item in enumerate(videos_to_process):
        assignments[idx % gpus_to_use].append(item)

    processes = []

    for gpu_id in range(gpus_to_use):
        p = mp.Process(
            target=worker,
            args=(gpu_id, assignments[gpu_id]),
        )
        p.start()
        processes.append(p)

    failed = False

    for p in processes:
        p.join()

        if p.exitcode != 0:
            failed = True
            print(f"Worker process failed with exit code {p.exitcode}", flush=True)

    if failed:
        raise RuntimeError("One or more GPU workers failed")

    print("Done", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
