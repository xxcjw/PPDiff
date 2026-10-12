import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _resolve_paths(args):
    args.image_dir = str(PROJECT_ROOT / args.image_dir)
    args.train_json = str(PROJECT_ROOT / args.train_json)
    args.test_json = str(PROJECT_ROOT / args.test_json)
    for p, name in [(args.image_dir, "image dir"),
                     (args.train_json, "train set"),
                     (args.test_json, "test set")]:
        if not Path(p).exists():
            logging.error(f"{name} does not exist: {p}")
            sys.exit(1)
    return args


def cmd_preprocess(args):
    from src.data.preprocess import main as preprocess_main
    preprocess_main(config_path=args.config, log_level=args.log_level,
                    skip_download=args.skip_download)


def cmd_profile(args):
    import yaml
    from src.user_profile.llm_client import LLMClient
    from src.user_profile.extractor import UserProfileExtractor

    config_path = str(PROJECT_ROOT / args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llm"]
    ext_cfg = config["extractor"]

    llm = LLMClient(
        model=llm_cfg["model"],
        mode=llm_cfg.get("mode", "vllm"),
        api_base=llm_cfg.get("api_base", "http://localhost:8800"),
        api_key=llm_cfg.get("api_key", ""),
        device=llm_cfg.get("device", "auto"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )

    extractor = UserProfileExtractor(
        llm_client=llm,
        item_info_path=str(PROJECT_ROOT / ext_cfg["item_info_path"]),
        beauty5_path=str(PROJECT_ROOT / ext_cfg["beauty5_path"]),
        output_dir=str(PROJECT_ROOT / ext_cfg["output_dir"]),
    )

    max_users = args.max_users if args.max_users else ext_cfg.get("max_users")
    extractor.extract_all(
        train_json=str(PROJECT_ROOT / ext_cfg["train_json"]),
        max_users=max_users,
    )


def cmd_gen_eval_styles(args):
    from src.user_profile.gen_eval_styles import main as gen_eval_styles_main

    argv = ["--config", args.config,
            "--test-json", args.test_json]
    if args.profiles_dir:
        argv += ["--profiles-dir", args.profiles_dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if getattr(args, "from_eval_config", None):
        argv += ["--from-eval-config", args.from_eval_config]
    if getattr(args, "all_test_users", False):
        argv.append("--all-test-users")
    if args.max_users:
        argv += ["--max-users", str(args.max_users)]
    if args.overwrite:
        argv.append("--overwrite")
    if args.dry_run:
        argv.append("--dry-run")
    return gen_eval_styles_main(argv)


def cmd_mask(args):
    from src.mask.extractor import MaskExtractor
    import json

    image_dir = str(PROJECT_ROOT / args.image_dir)
    mask_dir = str(PROJECT_ROOT / args.mask_dir)

    if args.item_json:
        item_ids = set()
        for json_file in args.item_json:
            with open(str(PROJECT_ROOT / json_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                item_ids.update(r["item"] for r in data)
            else:
                item_ids.update(data.keys())
        item_ids = list(item_ids)
    else:
        item_ids = None

    extractor = MaskExtractor(device=args.device)
    extractor.extract_and_save(image_dir, mask_dir, item_ids=item_ids)


def cmd_train(args):
    _resolve_paths(args)

    if args.version == "v1":
        from src.model.v1.trainer import Trainer, load_config
    else:
        from src.model.trainer import Trainer, load_config

    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train(
        image_dir=args.image_dir,
        train_json=args.train_json,
        test_json=args.test_json,
        profiles_dir=args.profiles_dir,
        mask_dir=args.mask_dir,
        resume_from=args.resume,
        max_users=args.max_users,
    )


def cmd_train_baseline(args):
    from src.baselines.train_baseline import main as baseline_train_main

    exit_code = baseline_train_main([
        "--config", args.config,
        "--model", args.model,
        "--device", args.device,
    ] + (["--epochs", str(args.epochs)] if args.epochs else []) +
        (["--output", args.output] if args.output else []))
    sys.exit(exit_code)


def cmd_gen_versions(args):
    from src.baselines.generate_versions import main as gen_versions_main

    argv = ["--config", args.config, "--device", args.device]
    if args.max_users:
        argv += ["--max-users", str(args.max_users)]
    if args.batch_size:
        argv += ["--batch-size", str(args.batch_size)]
    sys.exit(gen_versions_main(argv))


def cmd_eval_baseline(args):
    from src.experiment.eval_baseline_swap import main as eval_swap_main

    argv = ["--config", args.config, "--model", args.model, "--device", args.device]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if args.max_users:
        argv += ["--max-users", str(args.max_users)]
    if getattr(args, "feat_mode", None):
        argv += ["--feat-mode", args.feat_mode]
    sys.exit(eval_swap_main(argv))


def main():
    parser = argparse.ArgumentParser(
        description="IRec: Diffusion-based Personalized Image Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="available commands")

    p_pre = subparsers.add_parser("preprocess", help="data preprocessing")
    p_pre.add_argument("--config", default="config/Beauty/beauty.yaml")
    p_pre.add_argument("--skip-download", action="store_true")

    p_profile = subparsers.add_parser("profile", help="user profile extraction")
    p_profile.add_argument("--config", default="config/Beauty/profile.yaml")
    p_profile.add_argument("--max-users", type=int, default=None)

    p_ges = subparsers.add_parser(
        "gen-eval-styles",
        help="Generate visual presentation conditions for eval users' test targets (used by the evaluation script's profiles_dir)")
    p_ges.add_argument("--config", default="config/Beauty/profile.yaml")
    p_ges.add_argument("--profiles-dir", default=None,
                       help="Training-side profiles dir; defaults to extractor.output_dir from --config (do not mix across datasets)")
    p_ges.add_argument("--test-json", default="data/processed/Beauty/test.json")
    p_ges.add_argument("--output-dir", default=None,
                       help="Defaults to extractor.eval_output_dir in profile.yaml")
    p_ges.add_argument("--from-eval-config", default=None,
                       help="Path to baseline.yaml; defaults to the baseline.yaml next to --config, "
                            "generating only the num_eval_users users actually used by the evaluation pipeline")
    p_ges.add_argument("--all-test-users", action="store_true",
                       help="Explicitly generate for all users in test.json (off by default; evaluation only uses num_eval_users)")
    p_ges.add_argument("--max-users", type=int, default=None,
                       help="With --from-eval-config: number of eval users to select; otherwise: max users to generate")
    p_ges.add_argument("--overwrite", action="store_true")
    p_ges.add_argument("--dry-run", action="store_true")

    p_mask = subparsers.add_parser("mask", help="Extract subject masks for item images")
    p_mask.add_argument("--image-dir", default="pic/Beauty")
    p_mask.add_argument("--mask-dir", default="pic/Beauty_masks")
    p_mask.add_argument("--item-json", nargs="+", default=["data/processed/Beauty/train.json"],
                        help="Extract the item ID list from these files (multiple allowed)")
    p_mask.add_argument("--device", default="auto")

    p_train = subparsers.add_parser("train", help="model training")
    p_train.add_argument("--version", default="v2", choices=["v1", "v2"])
    p_train.add_argument("--config", default="config/Beauty/diffusion.yaml")
    p_train.add_argument("--image-dir", default="pic/Beauty")
    p_train.add_argument("--train-json", default="data/processed/Beauty/train.json")
    p_train.add_argument("--test-json", default="data/processed/Beauty/test.json")
    p_train.add_argument("--profiles-dir", default="data/processed/Beauty/profiles")
    p_train.add_argument("--mask-dir", default="pic/Beauty_masks")
    p_train.add_argument("--max-users", type=int, default=0, help="Limit the number of training users (0 = all)")
    p_train.add_argument("--resume", default=None)

    p_tb = subparsers.add_parser("train-baseline", help="Train image-aware baselines (VBPR/Visual-CF/...)")
    p_tb.add_argument("--config", default="config/Beauty/baseline.yaml")
    p_tb.add_argument("--model", default="vbpr",
                      choices=["vbpr", "visualcf", "freedom", "bm3", "diffmm",
                               "mentor"])
    p_tb.add_argument("--epochs", type=int, default=None)
    p_tb.add_argument("--device", default="auto")
    p_tb.add_argument("--output", default=None)

    p_gv = subparsers.add_parser("gen-versions", help="Generate 4 presentation variants of targets into gen_cache")
    p_gv.add_argument("--config", default="config/Beauty/baseline.yaml")
    p_gv.add_argument("--max-users", type=int, default=None)
    p_gv.add_argument("--batch-size", type=int, default=4)
    p_gv.add_argument("--device", default="auto")

    p_eb = subparsers.add_parser("eval-baseline", help="Run target-version swap evaluation with image-aware baselines")
    p_eb.add_argument("--config", default="config/Beauty/baseline.yaml")
    p_eb.add_argument("--model", default="vbpr",
                      choices=["vbpr", "visualcf", "freedom", "bm3", "diffmm",
                               "mentor"])
    p_eb.add_argument("--checkpoint", default=None)
    p_eb.add_argument("--max-users", type=int, default=None)
    p_eb.add_argument("--feat-mode", default=None, choices=["full", "bg"],
                      help="full = full-image features (default); bg = subject grayed out, background features only (amplifies personalization signal)")
    p_eb.add_argument("--device", default="auto")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"preprocess": cmd_preprocess, "profile": cmd_profile, "mask": cmd_mask,
     "gen-eval-styles": cmd_gen_eval_styles,
     "train": cmd_train,
     "train-baseline": cmd_train_baseline,
     "gen-versions": cmd_gen_versions,
     "eval-baseline": cmd_eval_baseline}[args.command](args)


if __name__ == "__main__":
    main()
