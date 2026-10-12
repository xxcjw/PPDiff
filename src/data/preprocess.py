import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_beauty5, load_config, load_meta, load_ratings
from src.data.data_split import compute_statistics, leave_one_out_split
from src.data.image_downloader import download_all_images
from src.data.sequence_builder import (
    build_user_sequences,
    build_user_sequences_with_review,
    filter_sequences,
    get_all_items,
)


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def save_json(data, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(config_path: str, log_level: str = "INFO", skip_download: bool = False):
    setup_logging(log_level)
    logger = logging.getLogger("preprocess")

    logger.info("=" * 60)
    logger.info("Step 1: Load config")
    logger.info("=" * 60)

    config = load_config(config_path)
    data_cfg = config["data"]
    download_cfg = config["download"]
    filter_cfg = config.get("filter", {})
    split_cfg = config["split"]

    raw_dir = PROJECT_ROOT / data_cfg["raw_data_dir"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    image_dir = PROJECT_ROOT / data_cfg["image_dir"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 2: Load Beauty_5 interactions (base data source)")
    logger.info("=" * 60)

    beauty5_path = str(raw_dir / data_cfg["beauty5_file"])
    beauty5_df = load_beauty5(beauty5_path)

    all_items_in_beauty5 = set(beauty5_df["item"].unique())
    logger.info(f"Beauty_5 has {beauty5_df['user'].nunique()} users, {len(all_items_in_beauty5)} items")

    max_users = filter_cfg.get("max_users", 2000)
    min_seq = filter_cfg.get("min_seq_len", 5)
    max_seq = filter_cfg.get("max_seq_len", 15)
    random_seed = filter_cfg.get("random_seed", 42)

    logger.info("=" * 60)
    logger.info(f"Step 2.5: Filter users (interactions {min_seq}-{max_seq}, at most {max_users})")
    logger.info("=" * 60)

    user_counts = beauty5_df.groupby("user").size()

    mask = (user_counts >= min_seq) & (user_counts <= max_seq)
    candidates = user_counts[mask].index.tolist()
    logger.info(f"Users with {min_seq}-{max_seq} interactions: {len(candidates)}")

    random.seed(random_seed)
    random.shuffle(candidates)
    selected_users = set(candidates[:max_users])

    beauty5_df = beauty5_df[beauty5_df["user"].isin(selected_users)].reset_index(drop=True)
    all_items_in_beauty5 = set(beauty5_df["item"].unique())

    logger.info(
        f"After filtering: {beauty5_df['user'].nunique()} users, "
        f"{len(all_items_in_beauty5)} items, "
        f"{len(beauty5_df)} interactions"
    )

    logger.info("=" * 60)
    logger.info("Step 3: Load meta data (filter to items in Beauty_5)")
    logger.info("=" * 60)

    meta_path = str(raw_dir / data_cfg["meta_file"])
    all_meta_dict = load_meta(meta_path)

    meta_dict = {}
    items_in_meta = 0
    items_missing_meta = 0
    for asin in all_items_in_beauty5:
        if asin in all_meta_dict:
            meta_dict[asin] = all_meta_dict[asin]
            items_in_meta += 1
        else:
            items_missing_meta += 1

    logger.info(
        f"Of {len(all_items_in_beauty5)} items after filtering, "
        f"{items_in_meta} have meta info, {items_missing_meta} missing"
    )

    item_info_path = str(processed_dir / data_cfg["output_item_info"])
    save_json(meta_dict, item_info_path)
    logger.info(f"Item info saved: {item_info_path}")

    logger.info("=" * 60)
    logger.info("Step 4: Build user interaction sequences from Beauty_5")
    logger.info("=" * 60)

    sequences = build_user_sequences(beauty5_df)
    sequences = filter_sequences(sequences, min_len=split_cfg.get("min_seq_len", 3))

    seq_path = str(processed_dir / data_cfg["output_sequences"])
    save_json(sequences, seq_path)
    logger.info(f"User interaction sequences saved: {seq_path}")

    logger.info("=" * 60)
    logger.info("Step 5: Download item images")
    logger.info("=" * 60)

    if skip_download:
        logger.info("Skipping image download (--skip-download)")
        download_stats = {"total": len(meta_dict), "success": 0, "failed": 0, "skipped": 0}
    else:
        download_stats = download_all_images(
            meta_dict=meta_dict,
            save_dir=str(image_dir),
            max_workers=download_cfg.get("max_workers", 4),
            max_retries=download_cfg.get("max_retries", 5),
            timeout=download_cfg.get("timeout", 15),
            retry_delay=download_cfg.get("retry_delay", 2.0),
        )

    logger.info("=" * 60)
    logger.info("Step 6: Remove items without images and re-filter sequences")
    logger.info("=" * 60)

    items_with_image = set()
    for asin in all_items_in_beauty5:
        for ext in ["jpg", "png", "gif", "webp"]:
            if (image_dir / f"{asin}.{ext}").exists():
                items_with_image.add(asin)
                break

    logger.info(f"Items with images: {len(items_with_image)} / {len(all_items_in_beauty5)}")

    before_users = len(sequences)
    before_interactions = sum(len(s) for s in sequences.values())
    removed_items_count = 0

    cleaned_sequences = {}
    for user, seq in sequences.items():
        cleaned_seq = [inter for inter in seq if inter["item"] in items_with_image]
        removed_items_count += len(seq) - len(cleaned_seq)
        if len(cleaned_seq) >= split_cfg.get("min_seq_len", 3):
            cleaned_sequences[user] = cleaned_seq

    after_users = len(cleaned_sequences)
    after_interactions = sum(len(s) for s in cleaned_sequences.values())

    logger.info(
        f"Cleanup done: removed {removed_items_count} interactions without images, "
        f"users {before_users} -> {after_users}, "
        f"interactions {before_interactions} -> {after_interactions}"
    )

    sequences = cleaned_sequences

    logger.info("=" * 60)
    logger.info("Step 7: Leave-One-Out split")
    logger.info("=" * 60)

    train_list, test_list = leave_one_out_split(sequences=sequences)

    train_path = str(processed_dir / data_cfg["output_train"])
    test_path = str(processed_dir / data_cfg["output_test"])

    save_json(train_list, train_path)
    save_json(test_list, test_path)
    logger.info(f"Train set saved: {train_path} ({len(train_list)} records)")
    logger.info(f"Test set saved: {test_path} ({len(test_list)} records)")

    logger.info("=" * 60)
    logger.info("Step 8: Generate statistics")
    logger.info("=" * 60)

    stats = compute_statistics(
        sequences=sequences,
        train_list=train_list,
        test_list=test_list,
        meta_dict=meta_dict,
        download_stats=download_stats,
    )

    stats_path = str(processed_dir / data_cfg["output_statistics"])
    save_json(stats, stats_path)
    logger.info(f"Statistics saved: {stats_path}")

    logger.info("=" * 60)
    logger.info("Preprocessing done!")
    logger.info("=" * 60)
    logger.info(f"Output dir: {processed_dir}")
    logger.info(f"Image dir: {image_dir}")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon Beauty data preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/beauty.yaml",
        help="Config file path (default: config/beauty.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the image download step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        log_level=args.log_level,
        skip_download=args.skip_download,
    )
