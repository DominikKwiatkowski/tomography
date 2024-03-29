import argparse
import wandb

from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.prepare_polar_training import prepare_polar_images
from src.testing.testing_config import TestingConfig
from src.testing.testing_manager import run_test
from src.training.training_config import TrainingConfig
from training.training_manager import run_training, RerunException
from src.utils import create_model, create_loss
import torch
import torch.multiprocessing as mp
import os
from datetime import datetime
from typing import IO
import sys


class FileConsoleOut:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def training_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--name", type=str, help="Batch size", required=True)
    parser.add_argument("--gpu", type=int, help="GPU no", required=True)
    parser.add_argument(
        "--img_size",
        type=int,
        help="Size of image, it should be lower than image width/height",
        required=True,
    )
    parser.add_argument("--metadata", type=str, help="Metadata path", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
    parser.add_argument(
        "--net_name",
        type=str,
        help="Network name",
        choices=["unet", "quanet", "defednet", "unetplusplus", "transformer"],
        required=True,
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        help="Loss name",
        choices=["dice", "cross_entropy", "mix"],
        required=True,
    )
    parser.add_argument("--tumor", action="store_true", help="Use tumor labels")
    parser.add_argument("--ww", type=int, help="Window width", default=2560)
    parser.add_argument("--wl", type=int, help="Window level", default=1536)
    parser.add_argument(
        "--discard", action="store_true", help="Discard images with 100% background"
    )
    parser.add_argument("--multiclass", action="store_true", help="Use multiclass")
    parser.add_argument("--use_polar", action="store_true", help="Use polar images")
    parser.add_argument("--seed", type=int, help="Seed of splitting images", default=42)
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = training_arg_parser()
    root_path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../")

    file_name = args.name
    # name is only filename without path
    name = f"{os.path.basename(file_name)}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.metadata)

    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(
        args.dataset,
        metadata,
        target_size=args.img_size,
        tumor=args.tumor,
        discard=args.discard,
        multiclass=args.multiclass,
        window_width=args.ww,
        window_center=args.wl,
    )

    if args.use_polar:
        model = create_model(args.net_name, args.multiclass, args.img_size).to(device)
        prepare_polar_images(
            dataset,
            name,
            model,
            args.batch_size,
            basic_dir=root_path,
            device=device,
            multiclass=args.multiclass,
        )
        name = f"{name}polar"
        file_name = f"{file_name}polar"

    _, test = dataset.train_val_test_k_fold(0.2, seed=args.seed)
    name = f"test{name}"
    dataset.test_mode = True
    test_dataset = dataset.create_data_loader(test, args.batch_size, seed=args.seed)

    model = create_model(args.net_name, args.multiclass, args.img_size).to(device)
    loss = create_loss(args.loss_name).to(device)

    test_config = TestingConfig(args.batch_size, args.multiclass, model, loss)
    wandb.init(project="master-thesis", entity="s175454", mode="online")
    wandb.config.update(args)
    wandb.run.name = f"{name}-{wandb.run.id}"
    run_test(file_name, test_config, device, test_dataset)


if __name__ == "__main__":
    main()
