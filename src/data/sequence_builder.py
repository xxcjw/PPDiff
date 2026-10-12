import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def build_user_sequences(
    ratings_df: pd.DataFrame,
) -> Dict[str, List[dict]]:
    logger.info("Building user interaction sequences...")

    sequences = {}
    for user, group in ratings_df.groupby("user"):
        group_sorted = group.sort_values("time")
        seq = []
        for _, row in group_sorted.iterrows():
            seq.append(
                {
                    "item": row["item"],
                    "rating": float(row["rating"]),
                    "time": int(row["time"]),
                }
            )
        sequences[user] = seq

    seq_lengths = [len(s) for s in sequences.values()]
    avg_len = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
    max_len = max(seq_lengths) if seq_lengths else 0
    min_len = min(seq_lengths) if seq_lengths else 0

    logger.info(
        f"Done: {len(sequences)} users, "
        f"avg sequence length {avg_len:.1f}, "
        f"max {max_len}, min {min_len}"
    )
    return sequences


def build_user_sequences_with_review(
    beauty5_df: pd.DataFrame,
) -> Dict[str, List[dict]]:
    logger.info("Building user interaction sequences with reviews...")

    sequences = {}
    for user, group in beauty5_df.groupby("user"):
        group_sorted = group.sort_values("time")
        seq = []
        for _, row in group_sorted.iterrows():
            seq.append(
                {
                    "item": row["item"],
                    "rating": float(row["rating"]),
                    "time": int(row["time"]),
                    "reviewText": row.get("reviewText", ""),
                    "summary": row.get("summary", ""),
                }
            )
        sequences[user] = seq

    logger.info(f"Done: {len(sequences)} users (with reviews)")
    return sequences


def filter_sequences(
    sequences: Dict[str, List[dict]],
    min_len: int = 2,
) -> Dict[str, List[dict]]:
    before_count = len(sequences)
    filtered = {
        user: seq for user, seq in sequences.items() if len(seq) >= min_len
    }
    after_count = len(filtered)

    logger.info(
        f"Filtering done (min_len={min_len}): "
        f"{before_count} -> {after_count} users, "
        f"removed {before_count - after_count} users"
    )
    return filtered


def get_all_items(
    sequences: Dict[str, List[dict]],
) -> set:
    items = set()
    for seq in sequences.values():
        for interaction in seq:
            items.add(interaction["item"])
    logger.info(f"{len(items)} unique items")
    return items
