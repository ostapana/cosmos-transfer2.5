import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from inception_metrics import MultiInceptionMetrics


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def list_videos(path):
    path = Path(path)
    return sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def load_video(path, frames=16, size=224, device="cuda"):
    reader = imageio.get_reader(str(path))
    video = []

    for frame in reader:
        x = torch.from_numpy(frame).float()
        if x.ndim == 2:
            x = x.unsqueeze(-1).repeat(1, 1, 3)
        if x.shape[-1] == 4:
            x = x[..., :3]
        video.append(x)

    if not video:
        raise RuntimeError(f"No frames read from {path}")

    video = torch.stack(video)

    if video.shape[0] < frames:
        pad = video[-1:].repeat(frames - video.shape[0], 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    idx = torch.linspace(0, video.shape[0] - 1, frames).long()
    video = video[idx]
    video = video.permute(0, 3, 1, 2)
    video = video / 127.5 - 1.0
    video = F.interpolate(video, size=(size, size), mode="bilinear", align_corners=False)

    return video.to(device)


def compute_metric(real_dir, fake_dir, metric_name, frames, batch_size, device, allow_unmatched=False):
    real_paths = list_videos(real_dir)
    fake_paths = list_videos(fake_dir)

    real_names = {p.name for p in real_paths}
    fake_names = {p.name for p in fake_paths}

    missing_in_fake = sorted(real_names - fake_names)
    extra_in_fake = sorted(fake_names - real_names)

    if (missing_in_fake or extra_in_fake) and not allow_unmatched:
        print(f"Real videos: {len(real_paths)}")
        print(f"Fake videos: {len(fake_paths)}")

        if missing_in_fake:
            print("\nMissing in fake:")
            for name in missing_in_fake:
                print(name)

        if extra_in_fake:
            print("\nExtra in fake:")
            for name in extra_in_fake:
                print(name)

        raise RuntimeError("Real/fake video filename mismatch")

    assert len(real_paths) >= 2, "Need at least 2 videos"

    if metric_name == "fvd":
        model = "i3d"
        size = 224
    elif metric_name == "fid":
        model = "inception"
        size = 299
    else:
        raise ValueError(metric_name)

    metric = MultiInceptionMetrics(
        device=device,
        compute_manifold=False,
        num_inception_chunks=10,
        manifold_k=3,
        model=model,
        use_precomputed_feat="",
    )

    with torch.no_grad():
        for i in range(0, len(real_paths), batch_size):
            real_batch = real_paths[i:i + batch_size]
            fake_batch = fake_paths[i:i + batch_size]

            if metric_name == "fvd":
                real = torch.stack([load_video(p, frames, size, device) for p in real_batch])
                fake = torch.stack([load_video(p, frames, size, device) for p in fake_batch])
            else:
                real = torch.cat([load_video(p, frames, size, device) for p in real_batch], dim=0)
                fake = torch.cat([load_video(p, frames, size, device) for p in fake_batch], dim=0)

            metric.update(real, image_type="real")
            metric.update(fake, image_type="fake")

    result = metric.compute()
    return {k: float(v) for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--metric", choices=["fid", "fvd"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-unmatched", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = compute_metric(
        real_dir=args.real,
        fake_dir=args.fake,
        metric_name=args.metric,
        frames=args.frames,
        batch_size=args.batch_size,
        device=device,
        allow_unmatched=args.allow_unmatched,
    )

    payload = {
        "name": args.name,
        "metric": args.metric.upper(),
        "real_dir": args.real,
        "fake_dir": args.fake,
        "frames": args.frames,
        "batch_size": args.batch_size,
        "device": device,
        "result": result,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
