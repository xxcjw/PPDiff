import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_profile.prompt import SYSTEM_ROLE, TARGET_ITEM_STYLE_PROMPT

logger = logging.getLogger("gen_eval_styles")

DESC_MAX_LEN = 500


def load_test_targets(test_json: str) -> dict:
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    targets = {}
    for r in test_data:
        targets.setdefault(r["user"], r["item"])
    logger.info(f"test.json: {len(test_data)} records, {len(targets)} users")
    return targets


def load_profile_texts(profiles_dir: str) -> dict:
    profiles = {}
    d = Path(profiles_dir)
    if not d.exists():
        raise FileNotFoundError(f"Training-side profiles dir does not exist: {d}")
    for p in sorted(d.glob("*.json")):
        if p.stem.startswith("_") or p.stem == "all_profiles":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read profile {p}: {e}")
            continue
        text = (data.get("profile") or "").strip()
        if text:
            profiles[p.stem] = text
    logger.info(f"Training-side profiles: profile text for {len(profiles)} users")
    return profiles


def build_prompt(profile: str, item_id: str, item_info: dict) -> str:
    meta = item_info.get(item_id, {})
    title = meta.get("title", "Unknown")
    description = meta.get("description", "No description") or "No description"
    if len(description) > DESC_MAX_LEN:
        description = description[:DESC_MAX_LEN] + "..."
    return TARGET_ITEM_STYLE_PROMPT.format(
        profile=profile, title=title, description=description,
    )


def diagnose_selection(all_targets: dict, profiles: dict, image_dir, mask_dir):
    from src.baselines.common import resolve
    from src.baselines.features import IMG_EXTENSIONS, find_image

    total = len(all_targets)
    no_profile = [u for u in all_targets if u not in profiles]
    with_profile = [u for u in all_targets if u in profiles]
    no_image = [u for u in with_profile if find_image(resolve(image_dir), all_targets[u]) is None]
    have_image = [u for u in with_profile if u not in set(no_image)]

    no_mask = []
    if mask_dir and Path(resolve(mask_dir)).exists():
        md = Path(resolve(mask_dir))
        no_mask = [u for u in have_image
                   if not any((md / f"{all_targets[u]}{e}").exists() for e in IMG_EXTENSIONS)]
    ok = len(have_image) - len(no_mask)

    logger.info(f"Selection diagnostics: test users {total} | no profile {len(no_profile)} | "
                f"has profile but target missing image {len(no_image)} | has image but target missing mask {len(no_mask)} | "
                f"all three present {ok}")
    if total and ok < total:
        logger.warning(f"Warning: Only {ok}/{total} test users satisfy the evaluation conditions; "
                       f"the final evaluated user count will be less than num_eval_users (an upper bound, not guaranteed)")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate visual presentation conditions for eval users' test targets")
    ap.add_argument("--config", default=None,
                    help="profile.yaml (provides llm and item_info_path)")
    ap.add_argument("--profiles-dir", default=None,
                    help="Training-side profiles dir (provides profile text); defaults to "
                         "extractor.output_dir in --config; never mix across datasets")
    ap.add_argument("--test-json", default="data/processed/Beauty/test.json")
    ap.add_argument("--item-info", default=None, help="Defaults to item_info_path in config")
    ap.add_argument("--output-dir", default=None,
                    help="Eval-side condition output dir (defaults to eval_output_dir in config)")
    ap.add_argument("--from-eval-config", default=None,
                    help="Path to baseline.yaml: reuse the evaluation pipeline's select_eval_users to pick the same "
                         "num_eval_users users (defaults to the baseline.yaml next to --config)")
    ap.add_argument("--all-test-users", action="store_true",
                    help="Explicitly generate for all users in test.json (off by default, because evaluation only uses "
                         "num_eval_users; the rest just wastes LLM calls)")
    ap.add_argument("--max-users", type=int, default=None,
                    help="With the eval config: how many eval users to select (overrides num_eval_users in config); "
                         "with --all-test-users: max number of users to generate")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate even if the output already exists")
    ap.add_argument("--dry-run", action="store_true", help="Only count pending users; do not call the LLM")
    args = ap.parse_args(argv)

    import yaml

    config_path = args.config or "config/Beauty/profile.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_cfg = config.get("llm", {})
    ext_cfg = config.get("extractor", {})

    item_info_path = args.item_info or ext_cfg.get(
        "item_info_path", "data/processed/Beauty/item_info.json")
    with open(item_info_path, "r", encoding="utf-8") as f:
        item_info = json.load(f)
    logger.info(f"item metadata: {len(item_info)}")

    profiles_dir = args.profiles_dir or ext_cfg.get("output_dir")
    if not profiles_dir:
        raise SystemExit(
            "Cannot determine the training-side profiles dir: specify it with --profiles-dir, "
            "or configure extractor.output_dir in profile.yaml (e.g. data/processed/Toys/profiles)")
    if not Path(profiles_dir).exists():
        raise SystemExit(f"Training-side profiles dir does not exist: {profiles_dir} (run `python src/main.py profile` first)")
    logger.info(f"Training-side profiles dir: {profiles_dir}")
    profiles = load_profile_texts(profiles_dir)

    eval_config = args.from_eval_config
    if not eval_config and not args.all_test_users:
        auto = Path(config_path).parent / "baseline.yaml"
        if auto.exists():
            eval_config = str(auto)
            logger.info(f"Auto-selecting users with the eval config: {eval_config}"
                        f" (add --all-test-users for all test users)")
        else:
            raise SystemExit(
                f"Eval config not found at {auto}: specify baseline.yaml with --from-eval-config; "
                f"if you really want conditions for all test.json users, add --all-test-users to confirm.")

    selection = None
    if eval_config:
        from src.baselines import common

        with open(eval_config, "r", encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f)
        d = eval_cfg.get("data", {})
        e = eval_cfg.get("eval", {})
        num_users = args.max_users or e.get("num_eval_users", 50)
        test_json = d.get("test_json", args.test_json)
        records = common.select_eval_users(

            test_json, d.get("image_dir"), profiles,
            mask_dir=d.get("mask_dir"), num_users=num_users)
        if not records:
            logger.error("select_eval_users selected no users (check test_json / images / masks)")
            return 1
        test_targets = {r["user"]: r["item"] for r in records}
        selection = {
            "mode": "select_eval_users",
            "eval_config": str(eval_config),
            "num_eval_users": num_users,
            "image_dir": d.get("image_dir"),
            "mask_dir": d.get("mask_dir"),
        }
        logger.info(f"Selected by the evaluation pipeline: {len(test_targets)} users (num_eval_users={num_users})")
        all_tt = load_test_targets(test_json)
        diagnose_selection(all_tt, profiles, d.get("image_dir"), d.get("mask_dir"))
        overlap = len(set(profiles) & set(all_tt))
        if all_tt and overlap < 0.5 * len(all_tt):
            logger.warning(
                f"Warning: Training-side profiles dir {profiles_dir} overlaps {test_json} users by only "
                f"{overlap}/{len(all_tt)}; the profiles dir may not match the dataset (mixed across datasets?). "
                f"Please check whether --profiles-dir / extractor.output_dir in profile.yaml is correct")
    else:
        test_targets = load_test_targets(args.test_json)
        logger.warning(f"Not using the eval config for selection: generating conditions for all {len(test_targets)} users in test.json, "
                       f"most of which will not be evaluated (drop --all-test-users to generate only eval users)")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif ext_cfg.get("eval_output_dir"):
        output_dir = Path(ext_cfg["eval_output_dir"])
    elif ext_cfg.get("output_dir"):
        output_dir = Path(ext_cfg["output_dir"]).with_name("profiles_eval")
        logger.info(f"No eval_output_dir in config; deriving the eval-side dir from the training-side profiles dir: {output_dir}")
    else:
        output_dir = Path("data/processed/Beauty/profiles_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    if selection and d.get("profiles_dir"):
        if Path(d["profiles_dir"]).as_posix().rstrip("/") != output_dir.as_posix().rstrip("/"):
            logger.warning(
                f"Warning: The eval config reads {d['profiles_dir']} but this run writes {output_dir} -- they differ, "
                f"so evaluation will not use the artifacts just generated. Change profiles_dir in baseline.yaml to "
                f"{output_dir}, or add --output-dir {d['profiles_dir']}")

    users = [u for u in test_targets if u in profiles]
    missing_profile = [u for u in test_targets if u not in profiles] if selection is None else []

    pending = [u for u in users if args.overwrite or not (output_dir / f"{u}.json").exists()]
    num_existing = len(users) - len(pending)
    todo = pending[: args.max_users] if args.max_users else pending

    logger.info(f"Users to generate: {len(users)} (skipping {len(missing_profile)} without profiles); "
                f"{num_existing} already exist; generating {len(todo)} this run")
    if args.dry_run:
        print(f"[dry-run] will generate {len(todo)} eval-side conditions -> {output_dir}")
        return 0
    if not todo:
        print(f"[skip] no users to generate ({num_existing} existing artifacts are reusable)")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable
    from src.user_profile.llm_client import LLMClient

    llm = LLMClient(
        model=llm_cfg.get("model", "Qwen3-4B-Instruct-2507"),
        mode=llm_cfg.get("mode", "vllm"),
        api_base=llm_cfg.get("api_base", "http://localhost:8800"),
        api_key=llm_cfg.get("api_key", ""),
        device=llm_cfg.get("device", "auto"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )

    done = 0
    for user in tqdm(todo, desc="Generating eval-side conditions", ncols=100):
        item_id = test_targets[user]
        prompt = build_prompt(profiles[user], item_id, item_info)
        try:
            style = llm.chat(prompt, system_prompt=SYSTEM_ROLE)
        except Exception as e:
            logger.error(f"{user} generation failed: {e}")
            continue
        if not style or not str(style).strip():
            logger.warning(f"{user} produced an empty result; skipping")
            continue
        record = {
            "user": user,
            "profile": profiles[user],
            "target_item": item_id,
            "target_style": style,
            "source": {
                "profiles_dir": str(profiles_dir),
                "test_json": str(args.test_json),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "protocol": "eval_side_C_{u,i_test}",
                "selection": selection,
            },
        }
        with open(output_dir / f"{user}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        done += 1

    summary = {
        "num_eval_users": len(users),
        "num_generated": done,
        "num_existing": num_existing,
        "num_skipped_no_profile": len(missing_profile),
        "profiles_dir": str(profiles_dir),
        "test_json": str(args.test_json),
        "output_dir": str(output_dir),
        "selection": selection,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "_gen_eval_styles_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Done: generated {done} new -> {output_dir}")
    logger.info(f"Point the eval config's profiles_dir to: {output_dir}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
