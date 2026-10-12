import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.unet import UNet
from src.model.noise_scheduler import DDPMScheduler, DDIMScheduler


class LatentDiffusion(nn.Module):

    def __init__(
        self,
        vae: nn.Module,
        unet: UNet,
        scheduler: DDPMScheduler,
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.prediction_type = prediction_type

        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)

    def compute_loss(
        self,
        images: torch.Tensor,
        user_condition: torch.Tensor = None,
    ) -> dict:
        device = images.device
        batch_size = images.shape[0]

        latents = self.encode_images(images)

        timesteps = self.scheduler.sample_timesteps(batch_size, device)

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, condition=user_condition)

        if not hasattr(self, '_debug_printed'):
            logger = logging.getLogger("ldm")
            logger.info(f"latents: {latents.shape}, noise_pred: {noise_pred.shape}")
            self._debug_printed = True

        if self.prediction_type == "epsilon":
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        elif self.prediction_type == "x0":
            loss = F.mse_loss(noise_pred, latents, reduction="mean")
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        return {"loss": loss, "mse": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        latent_size: tuple,
        user_condition: torch.Tensor = None,
        guidance_scale: float = 1.0,
        device: torch.device = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.unet.parameters()).device

        latents = torch.randn(num_samples, self.vae.latent_dim, *latent_size, device=device)

        if use_ddim:
            scheduler = DDIMScheduler(
                timesteps=self.scheduler.timesteps,
                beta_start=self.scheduler.betas[0].item(),
                beta_end=self.scheduler.betas[-1].item(),
            )
            step_size = self.scheduler.timesteps // ddim_steps
            timesteps = list(range(0, self.scheduler.timesteps, step_size))
        else:
            scheduler = self.scheduler
            timesteps = list(range(self.scheduler.timesteps))

        for t in reversed(timesteps):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            if guidance_scale > 1.0 and user_condition is not None:
                noise_cond = self.unet(latents, t_tensor, condition=user_condition)
                noise_uncond = self.unet(latents, t_tensor, condition=None)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            elif user_condition is not None:
                noise_pred = self.unet(latents, t_tensor, condition=user_condition)
            else:
                noise_pred = self.unet(latents, t_tensor, condition=None)

            latents = scheduler.step(noise_pred, t, latents)

        return self.decode_latents(latents)
