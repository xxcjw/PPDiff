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


def cmd_preprocess(args):
    from src.data.preprocess import main as preprocess_main
    preprocess_main(config_path=args.config, log_level=args.log_level,
                    skip_download=args.skip_download)


def cmd_train(args):
    from src.model.trainer import Trainer, load_config

    config = load_config(args.config)

    if args.image_dir:
        args.image_dir = str(PROJECT_ROOT / args.image_dir)
    if args.train_json:
        args.train_json = str(PROJECT_ROOT / args.train_json)
    if args.test_json:
        args.test_json = str(PROJECT_ROOT / args.test_json)

    for p, name in [(args.image_dir, "image dir"),
                     (args.train_json, "train set"),
                     (args.test_json, "test set")]:
        if not Path(p).exists():
            logging.error(f"{name} does not exist: {p}")
            sys.exit(1)

    trainer = Trainer(config)
    trainer.train(
        image_dir=args.image_dir,
        train_json=args.train_json,
        test_json=args.test_json,
        resume_from=args.resume,
    )


def cmd_evaluate(args):
    from src.model.trainer import Trainer, load_config

    config = load_config(args.config)
    args.image_dir = str(PROJECT_ROOT / args.image_dir)
    args.train_json = str(PROJECT_ROOT / args.train_json)
    args.test_json = str(PROJECT_ROOT / args.test_json)

    if not args.resume:
        logging.error("Evaluation mode requires --resume to load model weights")
        sys.exit(1)

    trainer = Trainer(config)
    item_id_to_idx = trainer._build_item_index(args.train_json)
    trainer._build_models(len(item_id_to_idx))
    trainer.load_checkpoint(args.resume)
    trainer.evaluate(args.image_dir, args.train_json, args.test_json)


def main():
    parser = argparse.ArgumentParser(
        description="IRec: Diffusion-based Personalized Image Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="available commands")

    p_pre = subparsers.add_parser("preprocess", help="data preprocessing")
    p_pre.add_argument("--config", default="config/beauty.yaml")
    p_pre.add_argument("--skip-download", action="store_true", help="Skip image download")

    p_train = subparsers.add_parser("train", help="model training")
    p_train.add_argument("--config", default="config/diffusion.yaml", help="diffusion model config")
    p_train.add_argument("--image-dir", default="pic/Beauty", help="image dir")
    p_train.add_argument("--train-json", default="data/processed/Beauty/train.json", help="train set")
    p_train.add_argument("--test-json", default="data/processed/Beauty/test.json", help="test set")
    p_train.add_argument("--resume", default=None, help="checkpoint path to resume training from")

    p_eval = subparsers.add_parser("evaluate", help="model evaluation")
    p_eval.add_argument("--config", default="config/diffusion.yaml")
    p_eval.add_argument("--image-dir", default="pic/Beauty")
    p_eval.add_argument("--train-json", default="data/processed/Beauty/train.json")
    p_eval.add_argument("--test-json", default="data/processed/Beauty/test.json")
    p_eval.add_argument("--resume", required=True, help="model checkpoint path")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
