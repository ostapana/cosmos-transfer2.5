"""Implementation of a depth to depth metric,
   where an extracted depth map from the original video is compared
   to the depth map of the reconstructed video through the normalized L1 depth error"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EPSILON = 1e-8

def read_depth_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {path}")

    return np.stack(frames)


def estimate_global_alignment(reference_frames, reconstructed_frames):
    """Estimate global scale and offset that align reconstructed depth values to the reference"""
    # Use up to 30 evenly spaced frames to estimate one alignment for the whole video
    sample_count = min(30, len(reference_frames))

    frame_indices = np.linspace(0, len(reference_frames) - 1, sample_count,).round().astype(int)

    random_generator = np.random.default_rng(0)

    reference_samples = []
    reconstructed_samples = []

    for frame_index in frame_indices:
        reference = reference_frames[frame_index].reshape(-1)
        reconstructed = reconstructed_frames[frame_index].reshape(-1)

        # Use at most 10,000 pixels from each selected frame
        selected_count = min(10000, reference.size)

        selected_indices = random_generator.choice(
            reference.size,
            size=selected_count,
            replace=False,
        )
        
        # Combine the sampled pixels from all selected frames
        reference_samples.append(reference[selected_indices])
        reconstructed_samples.append(reconstructed[selected_indices])

    reference_values = np.concatenate(reference_samples)
    reconstructed_values = np.concatenate(reconstructed_samples)

    # Build the linear model: reference ≈ scale * reconstructed + offset.
    matrix = np.column_stack([reconstructed_values, np.ones_like(reconstructed_values), ])

    # Estimate scale and offset using least-squares fitting.
    scale, offset = np.linalg.lstsq(matrix, reference_values, rcond=None, )[0]

    return float(scale), float(offset)


def calculate_median_depth_error(reference_frames, reconstructed_frames):
    if reference_frames.shape != reconstructed_frames.shape:
        raise ValueError(
            "Video shape mismatch: "
            f"reference={reference_frames.shape}, "
            f"reconstructed={reconstructed_frames.shape}"
        )

    scale, offset = estimate_global_alignment(reference_frames, reconstructed_frames)

    frame_errors = []

    for reference, reconstructed in zip(reference_frames, reconstructed_frames):
        # Align reconstructed depth values to the reference depth range
        reconstructed_aligned = scale * reconstructed + offset

        # Use a robust reference depth range that ignores extreme pixel values
        low = np.percentile(reference, 5)
        high = np.percentile(reference, 95)
        depth_range = high - low + EPSILON

        # Calculate normalized mean absolute error for the current frame
        normalized_error = (np.mean(np.abs(reconstructed_aligned - reference)) / depth_range)

        frame_errors.append(normalized_error)

    return float(np.median(frame_errors))

def evaluate_reconstructed_videos(reference_frames, reconstructed_paths):
    """Calculate and rank the depth error for reconstructed videos."""
    results = []

    for video_path in reconstructed_paths:
        reconstructed_frames = read_depth_video(video_path)

        median_error = calculate_median_depth_error(
            reference_frames,
            reconstructed_frames,
        )

        results.append(
            {
                "video": video_path.name,
                "median_depth_error": median_error,
            }
        )

    # sort results from the lowest median error to highest
    results.sort(key=lambda result: result["median_depth_error"])

    return results

def save_results(results, output_csv):
    """Save the ranked depth metric results to a CSV file"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["video", "median_depth_error"]

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-folder",
        type=Path,
        default=Path("data/depth_controls"),
    )

    parser.add_argument(
        "--reference-filename",
        type=str,
        default="day_depth.mp4",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/depth_controls/depth_video_comparison.csv"),
    )

    args = parser.parse_args()

    reference_path = args.input_folder / args.reference_filename

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference video not found: {reference_path}"
        )

    reconstructed_paths = (path for path in args.input_folder.glob("*.mp4")
                           if path.name != args.reference_filename)

    if not reconstructed_paths:
        raise RuntimeError(
            f"No reconstructed MP4 videos found in {args.input_folder}"
        )

    reference_frames = read_depth_video(reference_path)

    results = evaluate_reconstructed_videos(
        reference_frames,
        reconstructed_paths,
    )

    for position, result in enumerate(results, start=1):
        print(
            f"{position}. {result['video']}: "
            f"{result['median_depth_error']:.6f}"
        )

    save_results(results, args.output_csv)
