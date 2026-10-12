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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("trainer")


class RecTrainDataset(Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def __init__(self, train_json, image_dir, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
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
            user_sequences.setdefault(record["user"], []).append(record)

        self.samples = []
        for user, seq in user_sequences.items():
            if len(seq) < 2:
                continue
            sorted_seq = sorted(seq, key=lambda x: x["time"])
            last = sorted_seq[-1]
            img_path = self._find_image(last["item"])
            if img_path is None:
                continue
            self.samples.append({
                "user": user,
                "target_item": last["item"],
                "rating": last["rating"],
                "img_path": img_path,
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

        return image, sample["rating"], sample["user"], sample["target_item"]


def load_profiles(profiles_dir: str) -> dict:
    profiles_dir = Path(profiles_dir)
    profiles = {}
    if not profiles_dir.exists():
        logger.warning(f"Profiles dir does not exist: {profiles_dir}")
        return profiles

    for json_file in sorted(profiles_dir.glob("*.json")):
        user_id = json_file.stem
        if user_id == "all_profiles":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("target_style", "")
            if not text:
                text = data.get("profile", "")
            if text:
                profiles[user_id] = text
        except Exception as e:
            logger.warning(f"Failed to load profile {json_file}: {e}")

    logger.info(f"Loaded user profiles: {len(profiles)}")
    return profiles


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
        self.optimizer = None

    def _get_device(self, device_str):
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_models(self):
        import clip
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from src.model.latent_diffusion import LatentDiffusion

        cfg = self.config
        device = self.device
        sd_path = cfg["sd_model_path"]

        logger.info(f"Loading pretrained SD model: {sd_path}")

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float32).to(device)
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", torch_dtype=torch.float32).to(device)
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float32).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        noise_scheduler.config.num_train_timesteps = cfg["diffusion"]["timesteps"]

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        logger.info("CLIP ViT-B/32 loaded")

        freeze_strategy = cfg.get("unet", {}).get("freeze_strategy", "cross_attn_only")
        self.model = LatentDiffusion(
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            freeze_strategy=freeze_strategy,
        )

        self.model.to(device)

        trainable_params = list(p for p in self.model.unet.parameters() if p.requires_grad)
        if trainable_params:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=cfg["trainer"]["learning_rate"],
                weight_decay=cfg["trainer"].get("weight_decay", 0.0),
            )
        else:
            self.optimizer = None
            logger.info("No trainable parameters; skipping optimizer creation")

        trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {trainable:,}")

        self._pretrained_weights = {}
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad:
                self._pretrained_weights[name] = param.data.clone()

    def _compute_reg_loss(self, reg_weight):
        if not hasattr(self, '_pretrained_weights'):
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad and name in self._pretrained_weights:
                loss = loss + torch.nn.functional.mse_loss(param, self._pretrained_weights[name])
        return reg_weight * loss

    def _build_item_pool_features(self, image_dir: str) -> dict:
        import clip
        item_pool = {}
        for img_path in Path(image_dir).glob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                image = self.clip_preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.clip_model.encode_image(image).float()
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                item_pool[img_path.stem] = feat.squeeze(0)
            except Exception:
                continue
        logger.info(f"Item pool CLIP features: {len(item_pool)}")
        return item_pool

    def _decode_and_clip_encode(self, x0_pred):
        gen_images = self.model.decode_latents(x0_pred)
        gen_images = torch.clamp((gen_images + 1) / 2.0, 0, 1)

        gen_resized = torch.nn.functional.interpolate(gen_images, size=(224, 224), mode="bilinear")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        gen_clip_input = (gen_resized - mean) / std
        gen_features = self.clip_model.encode_image(gen_clip_input).float()
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        return gen_features

    def _compute_bpr_loss(self, gen_features, target_items, item_pool, item_id_list,
                          num_negatives=4):
        import random as rng

        item_features = torch.stack([item_pool[iid] for iid in item_id_list])
        item_id_to_idx = {iid: i for i, iid in enumerate(item_id_list)}

        batch_size = gen_features.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)

        for i in range(batch_size):
            target_id = target_items[i]
            if target_id not in item_id_to_idx:
                continue
            target_feat = item_features[item_id_to_idx[target_id]]
            pos_score = torch.dot(gen_features[i], target_feat)
            neg_ids = rng.sample(item_id_list, min(num_negatives, len(item_id_list) - 1))
            neg_ids = [nid for nid in neg_ids if nid != target_id][:num_negatives]
            for neg_id in neg_ids:
                neg_feat = item_features[item_id_to_idx[neg_id]]
                neg_score = torch.dot(gen_features[i], neg_feat)
                total_loss = total_loss - torch.nn.functional.logsigmoid(pos_score - neg_score)

        return total_loss / batch_size

    def _compute_style_loss(self, gen_features, profile_texts):
        import clip as clip_pkg

        text_tokens = clip_pkg.tokenize(profile_texts, truncate=True).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (gen_features * text_features).sum(dim=-1)
        return -similarity.mean()

    def save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        save_dict = {
            "epoch": epoch,
            "unet_state_dict": self.model.unet.state_dict(),
            "config": self.config,
        }
        if self.optimizer is not None:
            save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(save_dict, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.unet.load_state_dict(checkpoint["unet_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Checkpoint loaded: {path}, resuming from epoch {self.start_epoch}")

    @torch.no_grad()
    def direct_sample(self, images, text_embeddings, mask_bg=None,
                      guidance_scale=15.0, num_inference_steps=50, seeds=None):
        device = self.device
        B = images.shape[0]
        original_latents = self.model.encode_images(images)

        generators = None
        if seeds is not None:
            seeds = [int(s) for s in seeds]
            if len(seeds) != B:
                raise ValueError(f"seeds length {len(seeds)} != batch {B}")
            generators = []
            for s in seeds:
                g = torch.Generator(device=device)
                g.manual_seed(s)
                generators.append(g)
            shape_tail = tuple(original_latents.shape[1:])
            latents = torch.cat([
                torch.randn((1,) + shape_tail, generator=generators[i],
                            device=device, dtype=original_latents.dtype)
                for i in range(B)
            ])
        else:
            latents = torch.randn_like(original_latents)

        step_accepts_generator = False
        if generators is not None:
            try:
                import inspect
                step_accepts_generator = (
                    "generator" in inspect.signature(self.model.scheduler.step).parameters)
            except (TypeError, ValueError):
                step_accepts_generator = False
            if not step_accepts_generator:
                logger.warning("The current scheduler.step does not support the generator argument; "
                               "the scheduler's internal noise cannot be fixed per sample (initial noise is fixed)")

        uncond = self.model.encode_text([""] * B)
        cond = text_embeddings
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.model.scheduler.timesteps:
            if mask_bg is not None:
                latents = original_latents * (1 - mask_bg) + latents * mask_bg
            noise_uncond = self.model.unet(latents, t.expand(B), encoder_hidden_states=uncond).sample
            noise_cond = self.model.unet(latents, t.expand(B), encoder_hidden_states=cond).sample
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            if step_accepts_generator:
                latents = self.model.scheduler.step(
                    noise_pred, t, latents, generator=generators).prev_sample
            else:
                latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
        if mask_bg is not None:
            latents = original_latents * (1 - mask_bg) + latents * mask_bg
        return self.model.decode_latents(latents)

    def train(self, image_dir: str, train_json: str, test_json: str,
              profiles_dir: str = None, mask_dir: str = None, resume_from: str = None,
              max_users: int = 0):
        cfg = self.config

        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)

        profiles = {}
        if profiles_dir:
            profiles = load_profiles(profiles_dir)

        mask_handler = None
        if mask_dir and Path(mask_dir).exists():
            from src.mask.mask_handler import MaskHandler
            mask_handler = MaskHandler(mask_dir, downsample_factor=8)
            logger.info(f"Mask dir: {mask_dir}")

        image_size = cfg.get("image_size", 512)
        full_dataset = RecTrainDataset(
            train_json=train_json, image_dir=image_dir,
            image_size=image_size,
        )

        user_indices = []
        seen_users = set()
        limit = max_users if max_users > 0 else len(full_dataset)
        for i in range(len(full_dataset)):
            _, _, user, _ = full_dataset[i]
            if user in profiles and user not in seen_users:
                user_indices.append(i)
                seen_users.add(user)
                if len(seen_users) >= limit:
                    break

        if user_indices:
            train_dataset = Subset(full_dataset, user_indices)
            logger.info(f"Training dataset: {len(train_dataset)} users with profiles")
        else:
            train_dataset = full_dataset
            logger.info(f"Training dataset: {len(train_dataset)} users (unfiltered)")

        train_loader = DataLoader(
            train_dataset, batch_size=cfg["trainer"]["batch_size"],
            shuffle=True, num_workers=cfg["trainer"].get("num_workers", 2),
            pin_memory=True, drop_last=True,
        )

        import itertools

        grid_keys = []
        grid_values = []
        for k in ["diff_loss_weight", "style_loss_weight", "reg_loss_weight"]:
            v = cfg.get(k)
            if isinstance(v, list):
                grid_keys.append(k)
                grid_values.append(v)
        for k in ["epochs"]:
            v = cfg.get("trainer", {}).get(k)
            if isinstance(v, list):
                grid_keys.append(f"trainer.{k}")
                grid_values.append(v)

        if grid_keys:
            combinations = list(itertools.product(*grid_values))
            logger.info(f"Grid search: {len(combinations)} combinations, params: {grid_keys}")
        else:
            combinations = [()]

        base_output_dir = Path(cfg["trainer"]["output_dir"])
        base_ckpt_dir = Path(cfg["trainer"]["checkpoint_dir"])

        for combo_idx, combo in enumerate(combinations):
            for k, v in zip(grid_keys, combo):
                if k.startswith("trainer."):
                    cfg.setdefault("trainer", {})[k.split(".", 1)[1]] = v
                else:
                    cfg[k] = v

            if grid_keys:
                param_str = "_".join(f"{k}={v}" for k, v in zip(grid_keys, combo))
                run_dir = base_output_dir / f"run_{combo_idx+1:02d}_{param_str}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg["trainer"]["output_dir"] = str(run_dir)
                ckpt_dir = base_ckpt_dir / f"run_{combo_idx+1:02d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                cfg["trainer"]["checkpoint_dir"] = str(ckpt_dir)
                self.output_dir = run_dir
                self.checkpoint_dir = ckpt_dir
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"\n{'='*60}")
                logger.info(f"Run {combo_idx+1}/{len(combinations)}: {param_str}")
                logger.info(f"{'='*60}")

            self._build_models()

            item_pool = self._build_item_pool_features(image_dir)
            item_id_list = sorted(item_pool.keys())

            epochs = cfg["trainer"]["epochs"]
            if isinstance(epochs, list):
                epochs = epochs[0]
            rating_threshold = cfg.get("rating_threshold", 4.0)
            grad_clip = cfg["trainer"].get("grad_clip", 1.0)
            save_interval = cfg["trainer"].get("save_interval", 10)
            diff_loss_weight = cfg.get("diff_loss_weight", 0.5)
            style_loss_weight = cfg.get("style_loss_weight", 0.5)
            if isinstance(diff_loss_weight, list):
                diff_loss_weight = diff_loss_weight[0]
            if isinstance(style_loss_weight, list):
                style_loss_weight = style_loss_weight[0]

            logger.info(f"Loss: Diffusion({diff_loss_weight}) + Style({style_loss_weight})")

            for epoch in range(self.start_epoch, epochs):
                self.model.unet.train()
                epoch_loss = 0.0
                epoch_diff = 0.0
                epoch_style = 0.0
                num_batches = 0

                for images, ratings, users, target_items in train_loader:
                    images = images.to(self.device)
                    ratings = ratings.to(self.device)
                    text_embs = self._encode_user_profiles(users, ratings, profiles, rating_threshold)
                    profile_texts = []
                    use_threshold = cfg.get("use_rating_threshold", False)
                    for user, rating in zip(users, ratings):
                        if use_threshold:
                            if rating >= rating_threshold and user in profiles:
                                profile_texts.append(profiles[user])
                            else:
                                profile_texts.append("")
                        else:
                            if user in profiles:
                                profile_texts.append(profiles[user])
                            else:
                                profile_texts.append("")

                    mask_bg = None
                    if mask_handler is not None:
                        mask_bg = mask_handler.load_batch_masks(
                            list(target_items), image_size
                        ).to(self.device)

                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                    result = self.model.compute_loss(
                        images, text_embeddings=text_embs, mask_bg=mask_bg
                    )
                    x0_pred = result["x0_pred"]
                    diff_loss = result["loss_diff"]
                    gen_features = self._decode_and_clip_encode(x0_pred)
                    style_loss = self._compute_style_loss(gen_features, profile_texts)
                    reg_loss = self._compute_reg_loss(cfg.get("reg_loss_weight", 0.0))
                    total_loss = diff_loss_weight * diff_loss + style_loss_weight * style_loss + reg_loss

                    if self.optimizer is not None:
                        total_loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(p for p in self.model.unet.parameters() if p.requires_grad),
                                grad_clip,
                            )
                        self.optimizer.step()

                    epoch_loss += total_loss.item()
                    epoch_diff += diff_loss.item()
                    epoch_style += style_loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                avg_diff = epoch_diff / max(num_batches, 1)
                avg_style = epoch_style / max(num_batches, 1)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Total: {avg_loss:.6f}, Diffusion: {avg_diff:.6f}, Style: {avg_style:.6f}")

                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch + 1)

            self.save_checkpoint(epochs)
            logger.info(f"Run {combo_idx+1} training complete")

            gs_values = cfg.get("guidance_scale", 7.5)
            if not isinstance(gs_values, list):
                gs_values = [gs_values]

            self.model.eval()
            from torchvision.utils import save_image, make_grid

            for gs_val in gs_values:
                gs_dir = self.output_dir / f"guidance_{gs_val}"
                gs_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Sampling: guidance_scale={gs_val} -> {gs_dir}")

                batch_users = user_indices[:90]
                group_size = 6

                for g in range(15):
                    start = g * group_size
                    end = start + group_size
                    batch = batch_users[start:end]
                    if not batch:
                        break

                    all_originals = []
                    all_generated = []
                    for idx in batch:
                        img_tensor, rating, user_id, target_item = full_dataset[idx]
                        img_tensor = img_tensor.unsqueeze(0).to(self.device)
                        text_emb = self._encode_user_profiles(
                            [user_id], torch.tensor([rating]), profiles, rating_threshold
                        )
                        mask_bg = None
                        if mask_handler is not None:
                            mask_bg = mask_handler.load_batch_masks(
                                [target_item], image_size
                            ).to(self.device)

                        generated = self.direct_sample(
                            images=img_tensor, text_embeddings=text_emb,
                            mask_bg=mask_bg, guidance_scale=gs_val,
                        )
                        all_originals.append(img_tensor.cpu())
                        all_generated.append(generated.cpu())

                        gen_show = torch.clamp((generated + 1) / 2.0, 0, 1)
                        save_image(gen_show, gs_dir / f"gen_{g:02d}_{idx}.png")

                    originals = torch.clamp((torch.cat(all_originals) + 1) / 2.0, 0, 1)
                    generated = torch.clamp((torch.cat(all_generated) + 1) / 2.0, 0, 1)
                    comparison = torch.cat([originals, generated], dim=0)
                    grid = make_grid(comparison, nrow=group_size, padding=2)
                    save_image(grid, gs_dir / f"comparison_{g:02d}.png")

                logger.info(f"  guidance_scale={gs_val}: comparison images saved -> {gs_dir}")

            cfg["trainer"]["output_dir"] = str(base_output_dir)
            self.output_dir = base_output_dir
            cfg["trainer"]["checkpoint_dir"] = str(base_ckpt_dir)
            self.checkpoint_dir = base_ckpt_dir

        logger.info("Training complete!")

    def _encode_user_profiles(self, users, ratings, profiles, rating_threshold):
        use_threshold = self.config.get("use_rating_threshold", False)
        default_text = ""
        texts = []
        for user, rating in zip(users, ratings):
            if use_threshold:
                if rating >= rating_threshold and user in profiles:
                    texts.append(profiles[user])
                else:
                    texts.append(default_text)
            else:
                if user in profiles:
                    texts.append(profiles[user])
                else:
                    texts.append(default_text)

        return self.model.encode_text(texts)


def load_config(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
