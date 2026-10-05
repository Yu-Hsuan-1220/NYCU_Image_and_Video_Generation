from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear",
        betas: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if betas is not None:
            # A pretrained checkpoint already contains its exact noise schedule.
            betas = betas.detach().clone()
            if betas.ndim != 1 or betas.numel() != num_train_timesteps:
                raise ValueError("Saved betas must have one entry per training timestep.")
            if not torch.isfinite(betas).all() or not ((betas > 0) & (betas < 1)).all():
                raise ValueError("Saved betas must be finite and strictly between 0 and 1.")
        elif mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008
            # Build the T+1 endpoints of alpha_bar so that every beta_t is a
            # ratio of consecutive entries; this yields exactly T betas.
            ts = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            ts = ts / num_train_timesteps
            alphas_cumprod = torch.cos((ts + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0].clone()
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            # Clip beta_t to at most 0.999 (singularity at t = T).
            betas = betas.clamp(max=0.999)
               
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        
        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    
    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        # Normalise t once here so each step_predict_* below receives a 1-D
        # tensor on x_t's device, whether it was given a python int or the
        # 0-dim tensor sample() iterates over.
        if isinstance(t, int):
            t = torch.tensor([t])
        t = t.reshape(-1).to(x_t.device)

        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, alpha_bar_t, and alpha_bar_{t-1} from the
        #    scheduler (ᾱ_{t-1} = 1 at t = 0).
        # 2. Convert the predicted noise into the predicted clean sample
        #       x̂₀ = (x_t - √(1-ᾱ_t) * ε̂_θ) / √ᾱ_t
        #    and clamp it to [-1, 1].
        # 3. Compute the posterior mean
        #       \tilde{μ}_t = (√ᾱ_{t-1}·β_t/(1-ᾱ_t)) * x̂₀ + (√α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)) * x_t.
        # 4. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        # 5. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 6. Return the final sample at t-1.
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # alpha_bar_{t-1} is 1 at t = 0, so the last reverse step returns the
        # posterior mean exactly (a clamped index would give alpha_bar_0 here).
        is_first = (t == 0).reshape(-1, *([1] * (x_t.ndim - 1)))
        alpha_bar_t_prev = extract(self.alphas_cumprod, (t - 1).clamp(min=0), x_t)
        alpha_bar_t_prev = torch.where(
            is_first, torch.ones_like(alpha_bar_t_prev), alpha_bar_t_prev
        )

        # Invert the forward process to recover the predicted clean sample.
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_theta) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Posterior mean \tilde{mu}_t(x_t, x0_pred).
        mean = (
            alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t) * x0_pred
            + alpha_t.sqrt() * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x_t
        )

        # Posterior variance \tilde{beta}_t, which is 0 at t = 0.
        var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        # No noise is injected on the final step.
        noise = torch.randn_like(x_t)
        nonzero_mask = (~is_first).to(x_t.dtype)
        sample_prev = mean + nonzero_mask * var.sqrt() * noise

        #######################
        return sample_prev

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # Remember to clamp x0_pred to [-1, 1], as in step_predict_noise.
        # The network already outputs x0 directly, so unlike step_predict_noise
        # there is nothing to invert -- the same posterior formula is reused.
        x0_pred = x0_pred.clamp(-1, 1)

        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # alpha_bar_{t-1} is 1 at t = 0, so the last reverse step returns the
        # posterior mean exactly (a clamped index would give alpha_bar_0 here).
        is_first = (t == 0).reshape(-1, *([1] * (x_t.ndim - 1)))
        alpha_bar_t_prev = extract(self.alphas_cumprod, (t - 1).clamp(min=0), x_t)
        alpha_bar_t_prev = torch.where(
            is_first, torch.ones_like(alpha_bar_t_prev), alpha_bar_t_prev
        )

        # Posterior mean \tilde{mu}_t(x_t, x0_pred).
        mean = (
            alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t) * x0_pred
            + alpha_t.sqrt() * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x_t
        )

        # Posterior variance \tilde{beta}_t, which is 0 at t = 0.
        var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        # No noise is injected on the final step.
        noise = torch.randn_like(x_t)
        nonzero_mask = (~is_first).to(x_t.dtype)
        sample_prev = mean + nonzero_mask * var.sqrt() * noise

        #######################
        return sample_prev

    
    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # The network output *is* the posterior mean, so only the variance term
        # is still needed. No clamping: mean_theta is a mean, not an image.
        mean = mean_theta

        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # alpha_bar_{t-1} is 1 at t = 0, so the last reverse step returns the
        # posterior mean exactly (a clamped index would give alpha_bar_0 here).
        is_first = (t == 0).reshape(-1, *([1] * (x_t.ndim - 1)))
        alpha_bar_t_prev = extract(self.alphas_cumprod, (t - 1).clamp(min=0), x_t)
        alpha_bar_t_prev = torch.where(
            is_first, torch.ones_like(alpha_bar_t_prev), alpha_bar_t_prev
        )

        # Posterior variance \tilde{beta}_t, which is 0 at t = 0.
        var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        # No noise is injected on the final step.
        noise = torch.randn_like(x_t)
        nonzero_mask = (~is_first).to(x_t.dtype)
        sample_prev = mean + nonzero_mask * var.sqrt() * noise

        #######################
        return sample_prev

    
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn_like(x_0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        # q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0)
        x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps
        #######################

        return x_t, eps

class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode: str = "linear",
        num_inference_timesteps: int = 50,
        eta: float = 0.0,
        trained_scheduler: Optional[BaseScheduler] = None,
    ):
        if trained_scheduler is not None and num_train_timesteps != trained_scheduler.num_train_timesteps:
            raise ValueError("DDIM must use the checkpoint's number of training timesteps.")
        super().__init__(
            num_train_timesteps, beta_1, beta_T, mode,
            betas=None if trained_scheduler is None else trained_scheduler.betas,
        )
        if trained_scheduler is not None:
            # Copy before the timestep TODO, which may cache inference coefficients.
            self.alphas = trained_scheduler.alphas.detach().clone()
            self.alphas_cumprod = trained_scheduler.alphas_cumprod.detach().clone()
            self.schedule_mode = getattr(trained_scheduler, "schedule_mode", None)
        else:
            self.schedule_mode = mode
        self.eta = float(eta)
        self.set_inference_timesteps(num_inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Define the inference schedule (a subset of training timesteps, descending order).
        Inputs:
            num_inference_timesteps (int): number of inference steps (e.g., 50).
        """
        ######## TODO ########
        # Hint:
        #   - Define the DDIM inference schedule based on the given num_inference_timesteps.
        #   - The schedule should be a subset of training timesteps, ordered in descending fashion.
        #   - Store the result in `self.timesteps` (as a torch tensor) 
        #   - Store the step ratio in `self._ddim_step_ratio` for later use when computing previous t.
        #   - Compute a `step_ratio` that maps inference steps to training steps.
        # DO NOT change the code outside this part.
        self.num_inference_timesteps = num_inference_timesteps
        # Evenly spaced sub-sequence tau_1 < ... < tau_S of {0, ..., T-1}, e.g.
        # S=50, T=1000 -> step_ratio=20 and tau = [980, 960, ..., 20, 0].
        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (np.arange(0, num_inference_timesteps) * step_ratio).round()[::-1]
        self.timesteps = torch.from_numpy(timesteps.copy().astype(np.int64))
        self._ddim_step_ratio = step_ratio
        #######################

    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, predictor: str):
        """
        One step DDIM update: x_t -> x_{t_prev} with deterministic/stochastic control via eta.

        Input:
            x_t: [B,C,H,W]
            t: current absolute timestep index
            eps_theta: predicted noise
            predictor: predictor type
        Output:
            sample_prev: x at previous inference timestep
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        assert predictor == "noise", "In assignment 2, we only implement DDIM with noise predictor."
        if isinstance(t, int):
            t = torch.tensor([t])
        t = t.reshape(-1).to(x_t.device)
        t_prev = t - self._ddim_step_ratio

        # 1. alpha_bar at the current and previous inference timesteps;
        #    past the last step (t_prev < 0) alpha_bar_prev = 1, i.e. the clean sample.
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t)
        is_last = (t_prev < 0).reshape(-1, *([1] * (x_t.ndim - 1)))
        alpha_bar_prev = torch.where(is_last, torch.ones_like(alpha_bar_prev), alpha_bar_prev)

        # 2. Predicted clean sample x̂₀ = (x_t - √(1-ᾱ_t) ε̂_θ) / √ᾱ_t, clamped to the image range.
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_theta) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # 3. σ_t(η) = η √((1-ᾱ_prev)/(1-ᾱ_t)) √(1-ᾱ_t/ᾱ_prev)  (DDIM Eq. 16).
        #    η = 0: deterministic DDIM; η = 1: DDPM posterior variance.
        sigma = (
            self.eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
            * (1 - alpha_bar_t / alpha_bar_prev).sqrt()
        )

        # 4. Direction pointing to x_t: √(1-ᾱ_prev-σ²) ε̂_θ.
        dir_xt = (1 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * eps_theta

        # 5. x_{t_prev} = √ᾱ_prev x̂₀ + dir_xt + σ z  (DDIM Eq. 12).
        noise = torch.randn_like(x_t)
        sample_prev = alpha_bar_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        #######################
        return sample_prev
