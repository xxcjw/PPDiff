import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


class MaskExtractor:

    def __init__(self, device: str = "auto"):
        self.device = device
        self._session = None

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session("u2net")
            logger.info("rembg (u2net) loaded")
        return self._session

    def extract_mask(self, image_path: str) -> np.ndarray:
        from rembg import remove

        session = self._get_session()
        image = Image.open(image_path).convert("RGB")

        result = remove(image, session=session, only_mask=False)
        alpha = np.array(result.convert("RGBA").split()[-1])

        mask = 255 - alpha
        return mask

    def extract_and_save(
        self,
        image_dir: str,
        mask_dir: str,
        item_ids: List[str] = None,
    ):
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

        if item_ids:
            image_files = []
            for item_id in item_ids:
                for ext in IMG_EXTENSIONS:
                    path = image_dir / f"{item_id}{ext}"
                    if path.exists():
                        image_files.append(path)
                        break
        else:
            image_files = sorted([
                p for p in image_dir.iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            ])

        logger.info(f"Masks to extract: {len(image_files)} images")

        success = 0
        failed = 0
        for i, img_path in enumerate(image_files):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                success += 1
                continue

            try:
                mask = self.extract_mask(str(img_path))
                Image.fromarray(mask).save(str(mask_path))
                success += 1
            except Exception as e:
                logger.warning(f"Mask extraction failed [{img_path.name}]: {e}")
                failed += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Mask extraction progress: {i+1}/{len(image_files)}")

        logger.info(f"Mask extraction done: success {success}, failed {failed}")
