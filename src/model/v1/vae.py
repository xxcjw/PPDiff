import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PretrainedVAE(nn.Module):

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse", device: torch.device = None):
        super().__init__()
        from diffusers import AutoencoderKL

        logger.info(f"Loading pretrained VAE: {model_name}")
        self.vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float32)

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        self.latent_dim = 4
        self.latent_scale = 0.18215

        if device is not None:
            self.vae = self.vae.to(device)

        logger.info("Pretrained VAE loaded, weights frozen")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(x).latent_dist
        latent = posterior.mean * self.latent_scale
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent / self.latent_scale
        images = self.vae.decode(latent).sample
        return images

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class Encoder(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 4, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                nn.GroupNorm(32, h_dim),
                nn.SiLU(),
            ])
            in_channels = h_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        self.fc_log_var = nn.Conv2d(hidden_dims[-1], latent_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):

    def __init__(self, out_channels: int = 3, latent_dim: int = 4, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.input_conv = nn.Conv2d(latent_dim, hidden_dims[-1], 1)

        layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            layers.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1], 4, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dims[i - 1]),
                nn.SiLU(),
            ])

        layers.extend([
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], 4, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], out_channels, 3, padding=1),
            nn.Tanh(),
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.input_conv(z))


class VAE(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 4, hidden_dims: list = None):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(in_channels, latent_dim, hidden_dims)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return self.decoder(z), mu, log_var
