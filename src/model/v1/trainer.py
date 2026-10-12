import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.vae import PretrainedVAE, VAE
from src.model.unet import UNet
from src.model.noise_scheduler import DDPMScheduler
from src.model.latent_diffusion import LatentDiffusion
from src.model.user_profiler import UserProfiler

logger = logging.getLogger("trainer")


class RecTrainDataset(Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def __init__(self, train_json, image_dir, image_size=512,
                 item_id_to_idx=None, max_seq_len=20):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.item_id_to_idx = item_id_to_idx or {}
        self.max_seq_len = max_seq_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        user_sequences = {}
        for record in train_data:
            user = record["user"]
            if user not in user_sequences:
                user_sequences[user] = []
            user_sequences[user].append(record)

        self.samples = []
        for user, seq in user_sequences.items():
            if len(seq) < 2:
                continue
            last = seq[-1]
            img_path = self._find_image(last["item"])
            if img_path is None:
                continue

            history = seq[:-1]
            self.samples.append({
                "user": user,
                "target_item": last["item"],
                "rating": last["rating"],
                "img_path": img_path,
                "history": history,
            })

        logger.info(f"Training dataset: {len(self.samples)} user samples")

    def _find_image(self, item_id):
        for ext in self.IMG_EXTENSIONS:
            path = self.image_dir / f"{item_id}{ext}"
            if path.exists():
                return str(path)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample["img_path"]).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, self.image_size, self.image_size)

        rating = torch.tensor(sample["rating"], dtype=torch.float32)

        history = sample["history"]
        item_indices = []
        hist_ratings = []
        for h in history[-self.max_seq_len:]:
            item_idx = self.item_id_to_idx.get(h["item"], 0)
            item_indices.append(item_idx)
            hist_ratings.append(h["rating"])

        seq_len = len(item_indices)
        padding_len = self.max_seq_len - seq_len
        item_indices = [0] * padding_len + item_indices
        hist_ratings = [0.0] * padding_len + hist_ratings
        mask = [False] * padding_len + [True] * seq_len

        return (
            image,
            rating,
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(hist_ratings, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )


class RecTestDataset(Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def __init__(self, test_json, train_json, image_dir, image_size=512,
                 item_id_to_idx=None, max_seq_len=20):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.item_id_to_idx = item_id_to_idx or {}
        self.max_seq_len = max_seq_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        user_train_seq = {}
        for record in train_data:
            user = record["user"]
            if user not in user_train_seq:
                user_train_seq[user] = []
            user_train_seq[user].append(record)

        with open(test_json, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        self.samples = []
        for record in test_data:
            img_path = self._find_image(record["item"])
            if img_path is None:
                continue
            user = record["user"]
            history = user_train_seq.get(user, [])
            self.samples.append({
                "user": user,
                "target_item": record["item"],
                "img_path": img_path,
                "history": history,
            })

        logger.info(f"Test dataset: {len(self.samples)} user samples")

    def _find_image(self, item_id):
        for ext in self.IMG_EXTENSIONS:
            path = self.image_dir / f"{item_id}{ext}"
            if path.exists():
                return str(path)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["img_path"]).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, self.image_size, self.image_size)

        history = sample["history"]
        item_indices = []
        hist_ratings = []
        for h in history[-self.max_seq_len:]:
            item_indices.append(self.item_id_to_idx.get(h["item"], 0))
            hist_ratings.append(h["rating"])

        seq_len = len(item_indices)
        padding_len = self.max_seq_len - seq_len
        item_indices = [0] * padding_len + item_indices
        hist_ratings = [0.0] * padding_len + hist_ratings
        mask = [False] * padding_len + [True] * seq_len

        return (
            image,
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(hist_ratings, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            sample["user"],
            sample["target_item"],
        )


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.device = self._get_device(config["trainer"]["device"])

        seed = config["trainer"].get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.output_dir = Path(config["trainer"]["output_dir"])
        self.checkpoint_dir = Path(config["trainer"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0

        self.model = None
        self.profiler = None
        self.optimizer = None

    def _get_device(self, device_str):
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_item_index(self, train_json):
        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        all_items = sorted({r["item"] for r in train_data})
        item_id_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(all_items)}
        logger.info(f"Item index: {len(item_id_to_idx)} items")
        return item_id_to_idx

    def _build_models(self, num_items: int):
        cfg = self.config
        device = self.device

        self.profiler = UserProfiler(
            num_items=num_items,
            item_embedding_dim=cfg.get("item_embedding_dim", 128),
            user_embedding_dim=cfg["user_embedding_dim"],
            max_seq_len=cfg.get("max_seq_len", 20),
        ).to(device)

        vae_cfg = cfg.get("vae", {})
        if vae_cfg.get("pretrained", True):
            vae = PretrainedVAE(model_name=vae_cfg["model_name"], device=device)
        else:
            vae = VAE(
                in_channels=cfg["in_channels"],
                latent_dim=cfg["latent_channels"],
                hidden_dims=vae_cfg.get("hidden_dims", [64, 128, 256, 512]),
            ).to(device)

        unet = UNet(
            in_channels=cfg["latent_channels"],
            model_channels=cfg["unet"]["model_channels"],
            channel_mult=tuple(cfg["unet"]["channel_mult"]),
            num_res_blocks=cfg["unet"]["num_res_blocks"],
            attention_resolutions=tuple(cfg["unet"]["attention_resolutions"]),
            num_heads=cfg["unet"]["num_heads"],
            dropout=cfg["unet"].get("dropout", 0.0),
            condition_dim=cfg["user_embedding_dim"],
        ).to(device)

        scheduler = DDPMScheduler(
            timesteps=cfg["diffusion"]["timesteps"],
            beta_start=cfg["diffusion"]["beta_start"],
            beta_end=cfg["diffusion"]["beta_end"],
            beta_schedule=cfg["diffusion"].get("beta_schedule", "linear"),
        )

        self.model = LatentDiffusion(
            vae=vae, unet=unet, scheduler=scheduler,
            prediction_type=cfg["diffusion"].get("prediction_type", "epsilon"),
        )

        params = list(self.model.unet.parameters()) + list(self.profiler.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg["trainer"]["learning_rate"],
            weight_decay=cfg["trainer"].get("weight_decay", 0.0),
        )

        unet_params = sum(p.numel() for p in unet.parameters())
        profiler_params = sum(p.numel() for p in self.profiler.parameters())
        logger.info(f"UNet: {unet_params:,}, UserProfiler: {profiler_params:,}")

    def save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "unet_state_dict": self.model.unet.state_dict(),
            "profiler_state_dict": self.profiler.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.unet.load_state_dict(checkpoint["unet_state_dict"])
        self.profiler.load_state_dict(checkpoint["profiler_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Checkpoint loaded: {path}, resuming from epoch {self.start_epoch}")

    def train(self, image_dir: str, train_json: str, test_json: str,
              resume_from: str = None):
        cfg = self.config

        item_id_to_idx = self._build_item_index(train_json)

        self._build_models(num_items=len(item_id_to_idx))

        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)

        train_dataset = RecTrainDataset(
            train_json=train_json, image_dir=image_dir,
            image_size=cfg.get("image_size", 512),
            item_id_to_idx=item_id_to_idx,
            max_seq_len=cfg.get("max_seq_len", 20),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg["trainer"]["batch_size"],
            shuffle=True, num_workers=cfg["trainer"].get("num_workers", 2),
            pin_memory=True, drop_last=True,
        )

        epochs = cfg["trainer"]["epochs"]
        rating_threshold = cfg.get("rating_threshold", 4.0)
        grad_clip = cfg["trainer"].get("grad_clip", 1.0)
        save_interval = cfg["trainer"].get("save_interval", 10)
        sample_interval = cfg["trainer"].get("sample_interval", 5)
        image_size = cfg.get("image_size", 512)

        logger.info(f"Starting training: {epochs} epochs, device: {self.device}")
        logger.info(f"Rating threshold: {rating_threshold}")

        for epoch in range(self.start_epoch, epochs):
            self.model.unet.train()
            self.profiler.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, ratings, item_indices, hist_ratings, masks in train_loader:
                images = images.to(self.device)
                ratings = ratings.to(self.device)
                item_indices = item_indices.to(self.device)
                hist_ratings = hist_ratings.to(self.device)
                masks = masks.to(self.device)

                user_emb = self.profiler(item_indices, hist_ratings, masks)

                condition_mask = (ratings >= rating_threshold).float().unsqueeze(-1)
                condition = user_emb * condition_mask

                self.optimizer.zero_grad()
                losses = self.model.compute_loss(images, user_condition=condition)
                losses["loss"].backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.unet.parameters()) + list(self.profiler.parameters()),
                        grad_clip
                    )
                self.optimizer.step()

                epoch_loss += losses["mse"]
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)

            if (epoch + 1) % sample_interval == 0:
                self._generate_samples(epoch + 1, train_dataset, image_size)

        self.save_checkpoint(epochs)
        logger.info("Training complete!")

    @torch.no_grad()
    def _generate_samples(self, epoch, dataset, image_size):
        self.model.eval()
        self.profiler.eval()
        latent_size = (image_size // 8, image_size // 8)

        indices = random.sample(range(len(dataset)), min(4, len(dataset)))
        images, user_embs = [], []
        for idx in indices:
            img, rating, item_idx, hist_r, mask = dataset[idx]
            images.append(img)
            with torch.no_grad():
                emb = self.profiler(
                    item_idx.unsqueeze(0).to(self.device),
                    hist_r.unsqueeze(0).to(self.device),
                    mask.unsqueeze(0).to(self.device),
                )
                user_embs.append(emb.squeeze(0))

        images = torch.stack(images).to(self.device)
        user_embs = torch.stack(user_embs)

        generated = self.model.sample(
            num_samples=len(images), latent_size=latent_size,
            user_condition=user_embs, guidance_scale=2.0, device=self.device,
        )

        images_show = torch.clamp((images + 1) / 2.0, 0, 1)
        gen_show = torch.clamp((generated + 1) / 2.0, 0, 1)
        comparison = torch.cat([images_show, gen_show], dim=0)

        grid = make_grid(comparison, nrow=len(images), padding=2)
        save_path = self.output_dir / f"samples_epoch_{epoch:04d}.png"
        save_image(grid, save_path)
        logger.info(f"Samples saved: {save_path}")

    @torch.no_grad()
    def evaluate(self, image_dir: str, train_json: str, test_json: str,
                 topk: list = [5, 10]):
        self.model.eval()
        self.profiler.eval()
        cfg = self.config
        image_size = cfg.get("image_size", 512)
        latent_size = (image_size // 8, image_size // 8)
        guidance_scale = cfg.get("guidance_scale", 2.0)

        item_id_to_idx = self.profiler.item_id_to_idx

        test_dataset = RecTestDataset(
            test_json=test_json, train_json=train_json,
            image_dir=image_dir, image_size=image_size,
            item_id_to_idx=item_id_to_idx,
            max_seq_len=cfg.get("max_seq_len", 20),
        )

        logger.info("Building item pool...")
        item_pool = self._build_item_pool(image_dir)
        if item_pool is None:
            return

        hr = {k: 0 for k in topk}
        ndcg = {k: 0 for k in topk}
        total = 0

        for i in range(len(test_dataset)):
            image, item_idx, hist_r, mask, user_id, target_item = test_dataset[i]
            image = image.unsqueeze(0).to(self.device)

            user_emb = self.profiler(
                item_idx.unsqueeze(0).to(self.device),
                hist_r.unsqueeze(0).to(self.device),
                mask.unsqueeze(0).to(self.device),
            )

            gen_image = self.model.sample(
                num_samples=1, latent_size=latent_size,
                user_condition=user_emb, guidance_scale=guidance_scale,
                device=self.device,
            )

            gen_feat = self._extract_clip_feature(gen_image)
            if gen_feat is None:
                continue

            sims = {iid: torch.cosine_similarity(gen_feat, feat, dim=-1).item()
                    for iid, feat in item_pool.items()}
            ranked = [iid for iid, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)]

            if target_item in ranked:
                rank = ranked.index(target_item)
                for k in topk:
                    if rank < k:
                        hr[k] += 1
                        ndcg[k] += 1.0 / np.log2(rank + 2)
            total += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Evaluation progress: {i+1}/{len(test_dataset)}")

        logger.info("=" * 50)
        logger.info(f"Test evaluation results ({total} users)")
        for k in topk:
            logger.info(f"  HR@{k}: {hr[k]/max(total,1):.4f}, NDCG@{k}: {ndcg[k]/max(total,1):.4f}")
        logger.info("=" * 50)

        return {"hr": {k: hr[k]/max(total,1) for k in topk},
                "ndcg": {k: ndcg[k]/max(total,1) for k in topk}}

    def _build_item_pool(self, image_dir):
        try:
            import clip
            clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        except ImportError:
            logger.warning("clip is required")
            return None

        item_pool = {}
        for img_path in Path(image_dir).glob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = clip_model.encode_image(image).float()
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                item_pool[img_path.stem] = feat.squeeze(0)
            except Exception:
                continue
        logger.info(f"Item pool: {len(item_pool)} items")
        return item_pool

    def _extract_clip_feature(self, image):
        try:
            import clip
            clip_model, _ = clip.load("ViT-B/32", device=self.device)
        except ImportError:
            return None

        image = torch.clamp((image + 1) / 2.0, 0, 1)
        image = nn.functional.interpolate(image, size=(224, 224), mode="bilinear")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        image = (image - mean) / std

        with torch.no_grad():
            feat = clip_model.encode_image(image).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0)


def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_config(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Recommender diffusion model training")
    parser.add_argument("--config", type=str, default="config/diffusion.yaml")
    parser.add_argument("--image-dir", type=str, default="pic/Beauty")
    parser.add_argument("--train-json", type=str, default="data/processed/Beauty/train.json")
    parser.add_argument("--test-json", type=str, default="data/processed/Beauty/test.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    image_dir = str(PROJECT_ROOT / args.image_dir)
    train_json = str(PROJECT_ROOT / args.train_json)
    test_json = str(PROJECT_ROOT / args.test_json)

    for p, name in [(image_dir, "image dir"), (train_json, "train set"), (test_json, "test set")]:
        if not Path(p).exists():
            logger.error(f"{name} does not exist: {p}")
            sys.exit(1)

    trainer = Trainer(config)

    if args.eval_only:
        item_id_to_idx = trainer._build_item_index(train_json)
        trainer._build_models(len(item_id_to_idx))
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.evaluate(image_dir, train_json, test_json)
    else:
        trainer.train(image_dir, train_json, test_json, resume_from=args.resume)

if __name__ == "__main__":
    main()
