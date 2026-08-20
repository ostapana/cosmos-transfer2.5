# This script generates instance segmentation controls and per-instance masks for dataset videos
from pathlib import Path
from contextlib import nullcontext

import torch
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from utils import *

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO_DIR = ROOT_DIR / "data" / "video_dataset"
OUTPUT_SEGMENTATION_DIR = ROOT_DIR / "data" / "segmentation_controls"
PROMPT_PATH = ROOT_DIR / "data" / "segmentation_prompt.json"

print(f"Root dir: {ROOT_DIR}")
print(f"Input video dir: {INPUT_VIDEO_DIR}")

MODEL_NAME = "facebook/sam3"

TARGET_FPS = 10
TARGET_FRAMES = 100


def to_numpy(value):
    """Convert a tensor-like value to a NumPy array."""
    if torch.is_tensor(value):
        value = value.detach().cpu()

        if value.dtype == torch.bfloat16:
            value = value.float()

        return value.numpy()

    return np.asarray(value)


def frame_to_rgb(frame):
    """Convert an OpenCV BGR frame to an RGB image."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def update_object_prompts(frame_outputs, object_to_prompt):
    """Update the mapping between SAM3 object IDs and prompt classes."""
    prompt_mapping = frame_outputs.get("prompt_to_obj_ids", {})

    if not isinstance(prompt_mapping, dict):
        return

    for prompt, object_ids in prompt_mapping.items():
        for object_id in to_numpy(object_ids).reshape(-1):
            object_to_prompt[int(object_id)] = str(prompt)

def resize_mask(mask, width, height):
    """Resize a binary mask to the original video resolution."""
    if mask.shape == (height, width):
        return mask

    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def save_instance_masks(
    output_dir,
    frame_index,
    masks,
    object_ids,
    scores,
    object_to_prompt,
    class_to_id,
    width,
    height,
):
    """Save separate SAM3 instance masks and metadata for one video frame."""
    resized_masks = [
        resize_mask(mask > 0, width, height)
        for mask in masks
    ]

    if resized_masks:
        resized_masks = np.stack(resized_masks).astype(bool)
    else:
        resized_masks = np.zeros((0, height, width), dtype=bool)

    class_names = np.asarray(
        [
            object_to_prompt.get(int(object_id), "")
            for object_id in object_ids
        ],
        dtype=str,
    )
    class_ids = np.asarray(
        [class_to_id.get(class_name, 0) for class_name in class_names],
        dtype=np.int32,
    )

    np.savez_compressed(
        output_dir / f"{frame_index:05d}.npz",
        object_ids=object_ids.astype(np.int64),
        class_ids=class_ids,
        class_names=class_names,
        scores=scores.astype(np.float32),
        masks=resized_masks,
    )

    return resized_masks


def write_segmentation_video(
    model,
    processor,
    frames,
    prompts,
    output_path,
    width,
    height,
    device,
    dtype,
):
    """Run SAM3 and save instance-colored video plus separate instance masks."""
    rgb_frames = [frame_to_rgb(frame) for frame in frames]

    class_to_id = {
        prompt: class_id
        for class_id, prompt in enumerate(prompts, start=1)
    }

    inference_session = processor.init_video_session(
        video=rgb_frames,
        inference_device=device,
        inference_state_device="cpu",
        processing_device="cpu",
        video_storage_device="cpu",
        max_vision_features_cache_size=1,
        dtype=dtype,
    )

    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompts,
    )

    writer = create_video_writer(
        output_path,
        width,
        height,
        TARGET_FPS,
    )

    object_to_prompt = {}

    instance_output_dir = output_path.parent / f"{output_path.stem}_instances"
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), autocast_context:
        for frame_index, model_outputs in enumerate(model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(rgb_frames) - 1,
            show_progress_bar=True,
        )):
            frame_outputs = processor.postprocess_outputs(
                inference_session,
                model_outputs,
            )

            update_object_prompts(frame_outputs, object_to_prompt)

            masks = to_numpy(frame_outputs["masks"])
            object_ids = to_numpy(frame_outputs["object_ids"]).reshape(-1)
            scores = to_numpy(frame_outputs["scores"]).reshape(-1)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

            if masks.ndim == 2:
                masks = masks[None]

            # Save every instance as its own binary mask. Masks are not merged
            # even when multiple objects belong to the same semantic class.
            resized_masks = save_instance_masks(
                instance_output_dir,
                frame_index,
                masks,
                object_ids,
                scores,
                object_to_prompt,
                class_to_id,
                width,
                height,
            )

            segmentation = np.zeros(
                (height, width, 3),
                dtype=np.uint8,
            )

            # Give each SAM3 object ID its own deterministic visualization color.
            # Overlapping masks remain separate in the NPZ files; score only decides
            # which instance is visible on top in the rendered MP4.
            for mask_index in np.argsort(scores):
                object_id = int(object_ids[mask_index])
                class_name = object_to_prompt.get(object_id)

                if class_name not in class_to_id:
                    continue

                mask = resized_masks[mask_index]
                segmentation[mask] = get_class_color(object_id + 1)

            writer.write(cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

    writer.release()


def process_video(
    model,
    processor,
    prompts,
    video_path,
    output_path,
    device,
    dtype,
):
    """Estimate segmentation and save the visualization and instance masks."""
    frames, width, height = read_frames(video_path, TARGET_FRAMES)

    if frames is None:
        return

    print(f"Estimating segmentation: {video_path.name}", flush=True)

    write_segmentation_video(
        model,
        processor,
        frames,
        prompts,
        output_path,
        width,
        height,
        device,
        dtype,
    )

    instance_output_dir = output_path.parent / f"{output_path.stem}_instances"
    print(f"Saved {output_path}", flush=True)
    print(f"Saved instance masks to {instance_output_dir}", flush=True)


if __name__ == "__main__":
    # High-level wrapper for loading and running the segmentation model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    print(f"dtype: {dtype}")

    prompts = load_prompts(PROMPT_PATH)

    config = Sam3VideoConfig.from_pretrained(MODEL_NAME)
    config.score_threshold_detection = 0.35
    config.new_det_thresh = 0.50
    config.det_nms_thresh = 0.40
    config.assoc_iou_thresh = 0.10
    config.trk_assoc_iou_thresh = 0.35
    config.hotstart_unmatch_thresh = 15
    config.init_trk_keep_alive = 90
    config.max_trk_keep_alive = 90
    config.recondition_on_trk_masks = True
    config.recondition_every_nth_frame = 4
    config.high_conf_thresh = 0.65
    config.high_iou_thresh = 0.60
    config.suppress_overlapping_based_on_recent_occlusion_threshold = 0.85
    config.decrease_trk_keep_alive_for_empty_masklets = False
    config.fill_hole_area = 64

    model = Sam3VideoModel.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    processor = Sam3VideoProcessor.from_pretrained(MODEL_NAME)

    videos = get_videos_to_process(
        INPUT_VIDEO_DIR,
        OUTPUT_SEGMENTATION_DIR,
        "segmentation",
        required_dir_suffix="segmentation_instances",
    )

    for video_path, output_path in videos:
        process_video(
            model,
            processor,
            prompts,
            video_path,
            output_path,
            device,
            dtype,
        )

    print("Done", flush=True)
