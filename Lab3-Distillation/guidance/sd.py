from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "sd2-community/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) loss.

        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)

        Adds random noise to the latents at a random timestep and lets the
        diffusion model predict it. The difference between predicted and added
        noise is used as the gradient on the latents.

        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale

        Returns:
            loss: Scalar SDS loss
        """
        B = latents.shape[0]

        # Sample a random timestep t ~ U[min_step, max_step] and noise eps ~ N(0, I)
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        noise = torch.randn_like(latents)

        with torch.no_grad():
            # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)

        # SDS gradient with DreamFusion weighting w(t) = 1 - alpha_bar_t
        w = (1 - self.alphas[t]).view(B, 1, 1, 1)
        grad = torch.nan_to_num(w * (noise_pred - noise))

        # Reparameterize so that d(loss)/d(latents) == grad (skips the UNet Jacobian)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / B
        return loss

    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) loss.

        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)

        Same as SDS, but the added noise is replaced by the prediction of a
        LoRA-finetuned UNet that models the distribution of the current latents.
        The LoRA is trained jointly with a standard denoising loss.

        Args:
            lora_loss_weight: Weight of the LoRA training loss

        Returns:
            loss: Scalar VSD loss + lora_loss_weight * LoRA loss
        """
        B = latents.shape[0]
        cond_embeddings = text_embeddings.chunk(2)[1]

        # Sample a random timestep t ~ U[min_step, max_step] and noise eps ~ N(0, I)
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        noise = torch.randn_like(latents)

        with torch.no_grad():
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # eps_theta: pretrained UNet (LoRA disabled) with CFG
            self.unet.disable_adapters()
            noise_pred_pretrain = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)
            self.unet.enable_adapters()

            # eps_phi: LoRA UNet, conditional only (CFG scale 1)
            noise_pred_lora = self.unet(latents_noisy, t, encoder_hidden_states=cond_embeddings).sample

        # VSD gradient with weighting w(t) = 1 - alpha_bar_t
        w = (1 - self.alphas[t]).view(B, 1, 1, 1)
        grad = torch.nan_to_num(w * (noise_pred_pretrain - noise_pred_lora))

        # Reparameterize so that d(loss_vsd)/d(latents) == grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / B

        # LoRA loss: standard denoising loss on the (detached) current latents,
        # so eps_phi learns the distribution of the images being generated
        t_lora = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        noise_lora = torch.randn_like(latents)
        latents_noisy_lora = self.scheduler.add_noise(latents.detach(), noise_lora, t_lora)
        noise_pred_train = self.unet(latents_noisy_lora, t_lora, encoder_hidden_states=cond_embeddings).sample
        loss_lora = F.mse_loss(noise_pred_train.float(), noise_lora.float(), reduction="mean")

        return loss_vsd + lora_loss_weight * loss_lora

    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM inversion: x0 -> x_t.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Runs the DDIM update in reverse, from t=0 up to `target_t`, to find noisy
        latents consistent with x0. Write the loop yourself: you must NOT call any
        "invert" / "ddim_invert" / "invert_step" utilities from diffusers or other libraries.

        Args:
            target_t: Target timestep to invert to
            guidance_scale: CFG scale (typically negative for inversion)
            n_steps: Number of inversion steps
            eta: Stochasticity level (0 = deterministic)

        Returns:
            latents_noisy: Inverted noisy latents x_t (B, 4, H, W)
        """
        # Deterministic DDIM inversion (eta = 0): walk from t=0 up to target_t in n_steps
        B = latents.shape[0]
        target = int(target_t.max().item()) if torch.is_tensor(target_t) else int(target_t)
        timesteps = torch.linspace(0, target, n_steps + 1, device=self.device).round().long()

        latents_noisy = latents.clone()
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            alpha_cur = self.alphas[t_cur]
            alpha_next = self.alphas[t_next]

            noise_pred = self.get_noise_preds(latents_noisy, t_cur.expand(B), text_embeddings, guidance_scale)

            # x_{t_next} = sqrt(a_next) * x0_pred + sqrt(1 - a_next) * eps
            pred_x0 = (latents_noisy - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
            latents_noisy = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * noise_pred

        return latents_noisy

    def get_sdi_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
        current_iter=0,
        total_iters=500,
        inversion_guidance_scale=-7.5,
        inversion_n_steps=10,
        inversion_eta=0.3,
        update_interval=25,
    ):
        """
        Score Distillation via Inversion (SDI) loss.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Instead of random noise as in SDS, SDI gets the noise by DDIM inversion of
        the current latents. The timestep is annealed from max_step to min_step, and
        the denoised target is cached and refreshed every `update_interval` iterations.

        Args:
            current_iter: Current training iteration
            total_iters: Total number of training iterations
            inversion_guidance_scale, inversion_n_steps, inversion_eta:
                Passed to `invert_noise` as guidance_scale, n_steps, eta
            update_interval: Refresh the cached target every N iterations

        Returns:
            loss: Scalar MSE loss between current latents and the cached target
        """
        B = latents.shape[0]

        # Linearly anneal t from max_step down to min_step over the optimization
        progress = current_iter / max(1, total_iters)
        t_val = int(round(self.max_step - (self.max_step - self.min_step) * progress))
        t = torch.full((B,), t_val, dtype=torch.long, device=self.device)

        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')

        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )

                # Predict noise from the inverted noisy latents (with CFG)
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)

                # One-step denoise: x0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
                alpha_t = self.alphas[t].view(B, 1, 1, 1)
                target = (latents_noisy - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                target = torch.nan_to_num(target)

                # Cache the target
                self.sdi_target = target.detach()

        # MSE between current latents and the cached target
        loss = 0.5 * F.mse_loss(latents, self.sdi_target, reduction="sum") / B

        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
