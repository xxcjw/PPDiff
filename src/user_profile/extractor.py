import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_profile.prompt import (
    SYSTEM_ROLE,
    SINGLE_INTERACTION_PROMPT,
    AGGREGATE_PROMPT,
    STYLE_DESCRIPTION_PROMPT,
    TARGET_ITEM_STYLE_PROMPT,
)
from src.user_profile.llm_client import LLMClient

logger = logging.getLogger(__name__)


class UserProfileExtractor:

    def __init__(
        self,
        llm_client: LLMClient,
        item_info_path: str,
        beauty5_path: str,
        output_dir: str = "data/processed/Beauty/profiles",
    ):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(item_info_path, "r", encoding="utf-8") as f:
            self.item_info = json.load(f)
        logger.info(f"Loaded item metadata: {len(self.item_info)} items")

        self.review_data = {}
        with open(beauty5_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record.get("reviewerID", ""), record.get("asin", ""))
                self.review_data[key] = {
                    "reviewText": record.get("reviewText", ""),
                    "summary": record.get("summary", ""),
                    "overall": record.get("overall", 0),
                }
        logger.info(f"Loaded review data: {len(self.review_data)} records")

    def _build_user_histories(self, train_json: str):
        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        user_sequences = {}
        for record in train_data:
            user_sequences.setdefault(record["user"], []).append(record)

        user_histories = {}
        user_targets = {}
        for user, seq in user_sequences.items():
            if len(seq) < 2:
                continue
            sorted_seq = sorted(seq, key=lambda x: x["time"])
            target = sorted_seq[-1]
            history = sorted_seq[:-1]

            enriched = []
            for record in history:
                key = (user, record["item"])
                review = self.review_data.get(key, {})
                enriched.append({
                    **record,
                    "reviewText": review.get("reviewText", ""),
                    "summary": review.get("summary", ""),
                })
            user_histories[user] = enriched
            user_targets[user] = target

        logger.info(f"Built user histories: {len(user_histories)} users")
        return user_histories, user_targets

    def _format_interaction(self, record: dict) -> str:
        item_id = record["item"]
        item_meta = self.item_info.get(item_id, {})

        title = item_meta.get("title", "Unknown")
        description = item_meta.get("description", "No description")
        if len(description) > 500:
            description = description[:500] + "..."

        review = record.get("reviewText", "")
        if not review:
            review = record.get("summary", "No review")
        if len(review) > 500:
            review = review[:500] + "..."

        rating = record.get("rating", 0)

        return SINGLE_INTERACTION_PROMPT.format(
            title=title,
            description=description,
            review=review,
            rating=rating,
        )

    def extract_single_interaction(self, record: dict) -> str:
        prompt = self._format_interaction(record)
        return self.llm.chat(prompt, system_prompt=SYSTEM_ROLE)

    def extract_user_profile(self, user_id: str, history: List[dict], pbar=None) -> str:
        interaction_analyses = []
        for i, record in enumerate(history):
            if pbar:
                pbar.set_description(f"user {user_id[:8]}.. analyzing item {i+1}/{len(history)}")
            analysis = self.extract_single_interaction(record)
            item_id = record["item"]
            interaction_analyses.append(
                f"--- Item {i+1} ({item_id}) ---\n{analysis}"
            )

        if pbar:
            pbar.set_description(f"user {user_id[:8]}.. aggregating profile")
        analyses_text = "\n\n".join(interaction_analyses)
        aggregate_prompt = AGGREGATE_PROMPT.format(
            interaction_analyses=analyses_text
        )
        profile = self.llm.chat(aggregate_prompt, system_prompt=SYSTEM_ROLE)

        return profile

    def generate_target_item_style(self, profile: str, target_item_id: str) -> str:
        item_meta = self.item_info.get(target_item_id, {})
        title = item_meta.get("title", "Unknown")
        description = item_meta.get("description", "No description")
        if len(description) > 500:
            description = description[:500] + "..."

        prompt = TARGET_ITEM_STYLE_PROMPT.format(
            profile=profile,
            title=title,
            description=description,
        )
        return self.llm.chat(prompt, system_prompt=SYSTEM_ROLE)

    def extract_all(
        self,
        train_json: str,
        max_users: int = None,
    ) -> Dict[str, dict]:
        from tqdm import tqdm

        user_histories, user_targets = self._build_user_histories(train_json)

        if max_users:
            user_ids = list(user_histories.keys())[:max_users]
        else:
            user_ids = list(user_histories.keys())

        logger.info(f"Extracting profiles for {len(user_ids)} users")

        results = {}
        pbar = tqdm(user_ids, desc="Extracting user profiles", ncols=100)
        for user_id in pbar:
            history = user_histories[user_id]
            target = user_targets[user_id]
            target_item_id = target["item"]

            profile = self.extract_user_profile(user_id, history, pbar=pbar)

            pbar.set_description(f"user {user_id[:8]}.. generating target style")
            target_style = self.generate_target_item_style(profile, target_item_id)

            results[user_id] = {
                "profile": profile,
                "target_item": target_item_id,
                "target_style": target_style,
            }

            save_path = self.output_dir / f"{user_id}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results[user_id], f, ensure_ascii=False, indent=2)

        pbar.close()

        all_path = self.output_dir / "all_profiles.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Profile extraction done: {len(results)} users")
        logger.info(f"Output dir: {self.output_dir}")

        return results
