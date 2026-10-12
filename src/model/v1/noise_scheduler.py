import math
import torch
import numpy as np


class DDPMScheduler:

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.timesteps = timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).float()
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        device = x_0.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]

        while len(sqrt_alpha.shape) < len(x_0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.LongTensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        device = x_t.device
        beta_t = self.betas[t].to(device)
        alpha_t = self.alphas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)

        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        alpha_bar_prev = self.alphas_cumprod_prev[t].to(device)
        pred_mean = (
            torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * pred_x0
            + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
        )

        if t > 0:
            variance = self.posterior_variance[t].to(device)
            noise = torch.randn_like(x_t)
            return pred_mean + torch.sqrt(variance) * noise
        else:
            return pred_mean


class DDIMScheduler(DDPMScheduler):

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        device = x_t.device
        alpha_bar_t = self.alphas_cumprod[t].to(device)

        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if t > 0:
            alpha_bar_prev = self.alphas_cumprod_prev[t].to(device)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )

            pred_mean = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * model_output
            )

            noise = torch.randn_like(x_t)
            return pred_mean + sigma * noise
        else:
            return pred_x0
