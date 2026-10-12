import ast
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_meta(meta_path: str) -> Dict[str, dict]:
    logger.info(f"Loading meta data: {meta_path}")
    meta_dict = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = ast.literal_eval(line)
                asin = item.get("asin")
                if asin:
                    meta_dict[asin] = {
                        "asin": asin,
                        "title": item.get("title", ""),
                        "imUrl": item.get("imUrl", ""),
                        "description": item.get("description", ""),
                        "price": item.get("price"),
                        "brand": item.get("brand", ""),
                        "categories": item.get("categories", []),
                        "salesRank": item.get("salesRank", {}),
                        "related": item.get("related", {}),
                    }
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

    logger.info(f"Loaded meta info for {len(meta_dict)} items")
    return meta_dict


def load_ratings(ratings_path: str) -> pd.DataFrame:
    logger.info(f"Loading ratings data: {ratings_path}")

    df = pd.read_csv(
        ratings_path,
        header=None,
        names=["user", "item", "rating", "time"],
        dtype={"user": str, "item": str, "rating": float, "time": int},
    )

    df = df.sort_values("time").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} rating records, "
        f"{df['user'].nunique()} users, "
        f"{df['item'].nunique()} items"
    )
    return df


def load_beauty5(beauty5_path: str) -> pd.DataFrame:
    logger.info(f"Loading Beauty_5 data: {beauty5_path}")

    records = []
    with open(beauty5_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(
                    {
                        "user": record.get("reviewerID", ""),
                        "item": record.get("asin", ""),
                        "rating": record.get("overall", 0.0),
                        "time": record.get("unixReviewTime", 0),
                        "reviewText": record.get("reviewText", ""),
                        "summary": record.get("summary", ""),
                        "reviewerName": record.get("reviewerName", ""),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                continue

    df = pd.DataFrame(records)
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} reviews, "
        f"{df['user'].nunique()} users, "
        f"{df['item'].nunique()} items"
    )
    return df


def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded: {config_path}")
    return config
