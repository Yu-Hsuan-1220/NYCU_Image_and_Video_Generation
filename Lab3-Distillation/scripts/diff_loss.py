"""
Report-only experiment (Discussion / Fig. 6): optimize the latent with the original
diffusion loss L_Diff = ||eps_hat(x_t, y, t) - eps||^2, backpropagating through the UNet,
instead of SDS (which drops the UNet Jacobian). Same optimizer / schedule / clipping as main.py.

Usage: python scripts/diff_loss.py --device 1 --save_dir outputs/diff_loss
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from guidance.sd import StableDiffusion  # noqa: E402
from main import get_cosine_schedule_with_warmup  # noqa: E402
from utils import torch_to_pil  # noqa: E402

PROMPTS = ["A dog sitting on grass", "A castle next to a river", "A car on the road"]


def run(model, prompt, args):
    text_embeddings = torch.cat([model.get_text_embeds(args.negative_prompt), model.get_text_embeds(prompt)])
    latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=model.device))
    optimizer = torch.optim.AdamW([latents], lr=args.lr, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(args.steps * 1.5))

    for _ in tqdm(range(args.steps), desc=prompt):
        optimizer.zero_grad()
        t = torch.randint(model.min_step, model.max_step + 1, (1,), dtype=torch.long, device=model.device)
        noise = torch.randn_like(latents)
        latents_noisy = model.scheduler.add_noise(latents, noise, t)
        # Gradient flows through the UNet here (the term SDS omits)
        noise_pred = model.get_noise_preds(latents_noisy, t, text_embeddings, args.guidance_scale)
        loss = F.mse_loss(noise_pred, noise, reduction="sum")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latents], max_norm=1.0)
        optimizer.step()
        scheduler.step()

    img = model.decode_latents(latents.detach())
    torch_to_pil(img).save(os.path.join(args.save_dir, f"{prompt.replace(' ', '_')}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./outputs/diff_loss")
    parser.add_argument("--guidance_scale", type=float, default=25)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--negative_prompt", type=str, default="low quality")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model_args = argparse.Namespace(device=args.device, precision=torch.float32, loss_type="sds")
    model = StableDiffusion(model_args, t_range=[0.02, 0.98])
    model.unet.enable_gradient_checkpointing()
    for prompt in PROMPTS:
        run(model, prompt, args)


if __name__ == "__main__":
    main()
