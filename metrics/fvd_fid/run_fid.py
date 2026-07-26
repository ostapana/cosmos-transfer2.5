from pathlib import Path
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from inception_metrics import MultiInceptionMetrics


def load_video_frames(path, max_frames=16, size=299, device="cuda"):
    reader = imageio.get_reader(str(path))
    frames = []

    for frame in reader:
        x = torch.from_numpy(frame).float()
        if x.ndim == 2:
            x = x.unsqueeze(-1).repeat(1, 1, 3)
        if x.shape[-1] == 4:
            x = x[..., :3]
        frames.append(x)

    frames = torch.stack(frames)

    if frames.shape[0] < max_frames:
        pad = frames[-1:].repeat(max_frames - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)

    idx = torch.linspace(0, frames.shape[0] - 1, max_frames).long()
    frames = frames[idx]
    frames = frames.permute(0, 3, 1, 2)
    frames = frames / 127.5 - 1.0
    frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)

    return frames.to(device)


def compute_fid(real_dir, fake_dir, frames=16, batch_size=1, device="cuda"):
    real_paths = sorted(Path(real_dir).glob("*.mp4"))
    fake_paths = sorted(Path(fake_dir).glob("*.mp4"))

    assert len(real_paths) == len(fake_paths), f"{len(real_paths)} real, {len(fake_paths)} fake"
    assert len(real_paths) >= 2

    metric = MultiInceptionMetrics(
        device=device,
        compute_manifold=False,
        num_inception_chunks=10,
        manifold_k=3,
        model="inception",
        use_precomputed_feat="",
    )

    with torch.no_grad():
        for i in range(0, len(real_paths), batch_size):
            real_batch = real_paths[i:i + batch_size]
            fake_batch = fake_paths[i:i + batch_size]

            real = torch.cat([load_video_frames(p, frames, device=device) for p in real_batch], dim=0)
            fake = torch.cat([load_video_frames(p, frames, device=device) for p in fake_batch], dim=0)

            metric.update(real, image_type="real")
            metric.update(fake, image_type="fake")

    return metric.compute()


if __name__ == "__main__":
    result = compute_fid(
        real_dir="real",
        fake_dir="fake",
        frames=16,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(result)
