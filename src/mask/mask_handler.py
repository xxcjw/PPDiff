import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


class MaskHandler:

    def __init__(self, mask_dir: str, downsample_factor: int = 8):
        self.mask_dir = Path(mask_dir)
        self.downsample_factor = downsample_factor
        self._cache = {}

    def load_mask(self, item_id: str) -> torch.Tensor:
        if item_id in self._cache:
            return self._cache[item_id]

        mask_path = None
        for ext in IMG_EXTENSIONS:
            path = self.mask_dir / f"{item_id}{ext}"
            if path.exists():
                mask_path = path
                break

        if mask_path is None:
            return None

        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        self._cache[item_id] = mask_tensor
        return mask_tensor

    def to_latent_mask(self, mask: torch.Tensor, latent_size: tuple = None) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        if latent_size is None:
            H, W = mask.shape[2], mask.shape[3]
            latent_size = (H // self.downsample_factor, W // self.downsample_factor)

        latent_mask = F.interpolate(
            mask, size=latent_size, mode="bilinear", align_corners=False
        )

        latent_mask = (latent_mask > 0.5).float()

        return latent_mask

    def load_batch_masks(self, item_ids: list, image_size: int = 512) -> torch.Tensor:
        latent_size = (image_size // self.downsample_factor,
                       image_size // self.downsample_factor)
        latent_masks = []
        for item_id in item_ids:
            mask = self.load_mask(item_id)
            if mask is None:
                mask = torch.ones(1, image_size, image_size)
            latent_mask = self.to_latent_mask(mask, latent_size=latent_size)
            latent_masks.append(latent_mask)

        return torch.cat(latent_masks, dim=0)

    @staticmethod
    def composite_subject(
        original_latent: torch.Tensor,
        generated_latent: torch.Tensor,
        mask_bg: torch.Tensor,
    ) -> torch.Tensor:
        composited = original_latent * (1 - mask_bg) + generated_latent * mask_bg
        return composited
