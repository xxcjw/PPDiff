import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def leave_one_out_split(
    sequences: Dict[str, List[dict]],
    items_with_image: Set[str] = None,
) -> Tuple[List[dict], List[dict]]:
    logger.info("Running Leave-One-Out split...")

    train_list = []
    test_list = []
    skipped_users = 0

    for user, seq in sequences.items():
        if len(seq) < 2:
            skipped_users += 1
            continue

        test_interaction = seq[-1]
        train_interactions = seq[:-1]

        if items_with_image is not None:
            if test_interaction["item"] not in items_with_image:
                skipped_users += 1
                continue

        for interaction in train_interactions:
            train_list.append(
                {
                    "user": user,
                    "item": interaction["item"],
                    "rating": interaction["rating"],
                    "time": interaction["time"],
                }
            )

        test_list.append(
            {
                "user": user,
                "item": test_interaction["item"],
                "rating": test_interaction["rating"],
                "time": test_interaction["time"],
            }
        )

    logger.info(
        f"Split done: "
        f"train {len(train_list)} interactions, "
        f"test {len(test_list)} interactions, "
        f"skipped {skipped_users} users"
    )

    return train_list, test_list


def leave_one_out_split_with_review(
    sequences: Dict[str, List[dict]],
    items_with_image: Set[str] = None,
) -> Tuple[List[dict], List[dict]]:
    logger.info("Running Leave-One-Out split (with reviews)...")

    train_list = []
    test_list = []
    skipped_users = 0

    for user, seq in sequences.items():
        if len(seq) < 2:
            skipped_users += 1
            continue

        test_interaction = seq[-1]

        if items_with_image is not None:
            if test_interaction["item"] not in items_with_image:
                skipped_users += 1
                continue

        for interaction in seq[:-1]:
            train_list.append(
                {
                    "user": user,
                    "item": interaction["item"],
                    "rating": interaction["rating"],
                    "time": interaction["time"],
                    "reviewText": interaction.get("reviewText", ""),
                    "summary": interaction.get("summary", ""),
                }
            )

        test_list.append(
            {
                "user": user,
                "item": test_interaction["item"],
                "rating": test_interaction["rating"],
                "time": test_interaction["time"],
                "reviewText": test_interaction.get("reviewText", ""),
                "summary": test_interaction.get("summary", ""),
            }
        )

    logger.info(
        f"Split done: train {len(train_list)}, test {len(test_list)}"
    )
    return train_list, test_list


def get_train_items(train_list: List[dict]) -> set:
    return {item["item"] for item in train_list}


def get_test_users(test_list: List[dict]) -> set:
    return {item["user"] for item in test_list}


def compute_statistics(
    sequences: Dict[str, List[dict]],
    train_list: List[dict],
    test_list: List[dict],
    meta_dict: dict,
    download_stats: dict,
) -> dict:
    all_items = set()
    for seq in sequences.values():
        for interaction in seq:
            all_items.add(interaction["item"])

    num_users = len(sequences)
    num_items = len(all_items)
    num_interactions = sum(len(s) for s in sequences.values())
    seq_lengths = [len(s) for s in sequences.values()]

    train_users = {item["user"] for item in train_list}
    test_users = {item["user"] for item in test_list}
    users_in_both = train_users & test_users

    sparsity = 1.0 - num_interactions / (num_users * num_items) if num_users * num_items > 0 else 0

    train_items = get_train_items(train_list)
    train_items_with_image = sum(
        1 for item in train_items if item in meta_dict and meta_dict[item].get("imUrl")
    )

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "num_train_interactions": len(train_list),
        "num_test_interactions": len(test_list),
        "num_train_users": len(train_users),
        "num_test_users": len(test_users),
        "num_users_in_both": len(users_in_both),
        "avg_sequence_length": sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
        "max_sequence_length": max(seq_lengths) if seq_lengths else 0,
        "min_sequence_length": min(seq_lengths) if seq_lengths else 0,
        "sparsity": sparsity,
        "num_items_in_meta": len(meta_dict),
        "num_train_items_with_image": train_items_with_image,
        "image_download": download_stats,
    }

    logger.info("=== Dataset statistics ===")
    logger.info(f"  Users: {stats['num_users']}")
    logger.info(f"  Items: {stats['num_items']}")
    logger.info(f"  Total interactions: {stats['num_interactions']}")
    logger.info(f"  Train interactions: {stats['num_train_interactions']}")
    logger.info(f"  Test interactions: {stats['num_test_interactions']}")
    logger.info(f"  Avg sequence length: {stats['avg_sequence_length']:.1f}")
    logger.info(f"  Sparsity: {stats['sparsity']:.4f}")
    logger.info(f"  Image download: success {download_stats.get('success', 0)}, failed {download_stats.get('failed', 0)}")

    return stats
