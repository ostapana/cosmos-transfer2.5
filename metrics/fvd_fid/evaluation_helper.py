import numpy as np
import imageio.v2 as imageio
from einops import rearrange
from tqdm import tqdm

import torch

from Dataset.dataloader import get_data
from Metrics.inception_metrics import MultiInceptionMetrics
from Metrics.pixel_metrics import PixelMetrics


class EvaluationHelper:
    def __init__(self, trainer, generation_helper):
        self.fm = trainer
        self.generation_helper = generation_helper

    @torch.no_grad()
    def compute_score(self, num_video=1_000, num_step=50, tot_frames=16, save_exemple=False, compute_pr=False, use_precomputed_feat="", save_sample=False):
        video_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="i3d",
            use_precomputed_feat=use_precomputed_feat,
        )

        images_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="inception",
            use_precomputed_feat=use_precomputed_feat,
        )

        pixel_metrics = PixelMetrics(device=self.fm.args.device)

        eval_data = get_data(
            data=self.fm.args.data.split("_")[0],
            img_size=self.fm.args.img_size,
            n_frames=tot_frames,
            seed=self.fm.args.seed,
            data_folder=self.fm.args.eval_folder,
            cameras=self.fm.args.cameras,
            bsize=max(2, self.fm.args.bsize),
            num_workers=max(2, self.fm.args.bsize),
            is_multi_gpus=self.fm.args.is_multi_gpus
        )[0]

        total_videos = num_video // (self.fm.args.nb_gpus * self.fm.args.nb_cam)
        bar = tqdm(total=total_videos, leave=False, desc="Metrics Evaluation") if self.fm.args.is_master else None
        rollout_steps = (tot_frames - 1) // self.fm.args.n_frames + 1
        with self.fm.ema_scope():
            for i, batch in enumerate(eval_data):
                real_sample = batch["images"].to(self.fm.args.device)
                b, c, t, h, w_total = real_sample.size()
                w = w_total // self.fm.args.nb_cam

                gen_sample = self.generation_helper.generate_samples(
                    x_ctx=real_sample,
                    nb_video=b,
                    latent_context=1,
                    num_steps=num_step,
                    text=None,
                    cfg_w=0,
                    alpha=0.0,
                    scheduler_mode=self.fm.args.scheduler_mode,
                    rollout_steps=rollout_steps,
                    use_clip=False,
                )

                gen_sample = gen_sample[:, :, :tot_frames, :, :]
                gen_sample = rearrange(gen_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                gen_sample = torch.clamp(gen_sample, -1, 1).contiguous().float()

                real_sample = rearrange(real_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                real_sample = torch.clamp(((real_sample.float() / 255.0) * 2) - 1, -1, 1).contiguous().float()

                video_metrics.update(gen_sample, image_type="fake")
                video_metrics.update(real_sample, image_type="real")

                images_metrics.update(gen_sample, image_type="fake")
                images_metrics.update(real_sample, image_type="real")

                pixel_metrics.update(gen_sample, real_sample)

                if save_sample and i == 0 and self.fm.args.is_master:
                    video = torch.cat([gen_sample[0], real_sample[0]], dim=-1)
                    video = video.permute(0, 2, 3, 1).cpu().numpy()
                    video = ((video + 1) * 127.5).astype(np.uint8)
                    imageio.mimsave("./saved_video/generated_video.mp4", video, fps=10)

                if self.fm.args.is_master:
                    bar.update(b)

                if video_metrics.count >= total_videos:
                    break

        video_m = video_metrics.compute()
        images_m = images_metrics.compute()
        pixel_m = pixel_metrics.compute()

        metrics = {**video_m, **images_m, **pixel_m}
        metrics = {f"{k}": round(v, 4) for k, v in metrics.items()}

        if self.fm.args.is_master:
            bar.close()
            print(metrics)
            if save_exemple:
                name = str(self.fm.sampler).replace(" ", "_").replace(",", "").replace(":", "")
                with open(f"./results/" + name, "w") as file:
                    file.write(str(metrics))

        # Drop references before empty_cache to release GPU tensors promptly.
        batch = real_sample = gen_sample = video = None
        video_m = images_m = pixel_m = None
        eval_data = video_metrics = images_metrics = pixel_metrics = None

        if self.fm.args.device != "cpu":
            torch.cuda.empty_cache()

        return metrics
