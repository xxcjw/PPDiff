import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

logger = logging.getLogger(__name__)


class LatentDiffusion(nn.Module):

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder,
        tokenizer,
        scheduler: DDPMScheduler,
        freeze_strategy: str = "cross_attn_only",
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.freeze_strategy = freeze_strategy

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        self._freeze_unet(freeze_strategy)

    def _freeze_unet(self, strategy: str):
        if strategy == "cross_attn_only":
            for param in self.unet.parameters():
                param.requires_grad = False
            for name, param in self.unet.named_parameters():
                if "attn2" in name:
                    param.requires_grad = True
            logger.info("UNet: fine-tuning cross-attention layers only")
        elif strategy == "full":
            logger.info("UNet: fully fine-tuned")
        elif strategy == "none":
            for param in self.unet.parameters():
                param.requires_grad = False
            logger.info("UNet: fully frozen")

        trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.unet.parameters())
        logger.info(f"UNet params: {trainable:,} / {total:,} trainable")

    def encode_text(self, texts: list) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        inputs = self.tokenizer(
            texts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(inputs.input_ids)[0]
        return text_embeddings

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(images).latent_dist
        return posterior.mean * 0.18215

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215
        return self.vae.decode(latents).sample

    def compute_loss(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor = None,
        mask_bg: torch.Tensor = None,
    ) -> dict:
        device = images.device
        batch_size = images.shape[0]

        original_latents = self.encode_images(images)

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long,
        )

        noise = torch.randn_like(original_latents)
        noisy_latents = self.scheduler.add_noise(original_latents, noise, timesteps)

        if text_embeddings is not None:
            encoder_hidden_states = text_embeddings
        else:
            encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=device)

        model_output = self.unet(
            noisy_latents, timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss_diff = F.mse_loss(model_output, noise, reduction="mean")

        alpha_t = self.scheduler.alphas_cumprod.to(device)[timesteps]
        while len(alpha_t.shape) < len(noisy_latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        x0_pred = (noisy_latents - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)

        if mask_bg is not None:
            mask_latent = mask_bg.expand_as(original_latents)
            if not hasattr(self, '_mask_debug_printed'):
                self._mask_debug_printed = True
                logger.info(f"Mask debug: shape={mask_bg.shape}, "
                            f"range=[{mask_bg.min():.2f}, {mask_bg.max():.2f}], "
                            f"mean={mask_bg.mean():.2f}")
            x0_pred = original_latents * (1 - mask_latent) + x0_pred * mask_latent

        return {"x0_pred": x0_pred, "loss_diff": loss_diff}

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor = None,
        text_embeddings: torch.Tensor = None,
        mask_bg: torch.Tensor = None,
        guidance_scale: float = 7.5,
        device: torch.device = None,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.unet.parameters()).device

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if images is not None:
            original_latents = self.encode_images(images)
            batch_size = images.shape[0]
        else:
            original_latents = None
            batch_size = 1

        latents = torch.randn(batch_size, 4, 64, 64, device=device)

        uncond_hidden = torch.zeros(batch_size, 77, 768, device=device)
        if text_embeddings is not None:
            cond_hidden = text_embeddings
        else:
            cond_hidden = torch.zeros(batch_size, 77, 768, device=device)

        for t in self.scheduler.timesteps:
            if mask_bg is not None and original_latents is not None:
                latents = original_latents * (1 - mask_bg) + latents * mask_bg

            noise_uncond = self.unet(
                latents, t.expand(batch_size),
                encoder_hidden_states=uncond_hidden,
            ).sample

            noise_cond = self.unet(
                latents, t.expand(batch_size),
                encoder_hidden_states=cond_hidden,
            ).sample

            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if mask_bg is not None and original_latents is not None:
            latents = original_latents * (1 - mask_bg) + latents * mask_bg

        return self.decode_latents(latents)
