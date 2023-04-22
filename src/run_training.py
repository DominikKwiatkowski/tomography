import argparse
import wandb

from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
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
from prepare_polar_training import prepare_polar_images


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
    parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
    parser.add_argument("--gpu", type=int, help="GPU no", required=True)
    parser.add_argument("--fold", type=int, help="Fold number", required=True)
    parser.add_argument(
        "--learning_rate", type=float, help="network learning rate", required=True
    )
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
    parser.add_argument("--early_stop", action="store_true", help="Use early stop")
    parser.add_argument("--no_val", action="store_true", help="Remove validation set")
    parser.add_argument("--use_polar", action="store_true", help="Use polar images")
    parser.add_argument("--seed", type=int, help="Seed of splitting images", default=42)
    parser.add_argument("--optimizer", type=str, help="Optimizer", default="Adam")
    parser.add_argument(
        "--scheduler", type=str, help="Scheduler", default="ReduceLROnPlateau"
    )
    parser.add_argument(
        "--n_layers", type=int, help="Number of layers in transformer", default=12
    )
    parser.add_argument(
        "--n_heads", type=int, help="Number of heads in transformer", default=12
    )
    parser.add_argument(
        "--d_model", type=int, help="Dimension of model in transformer", default=768
    )
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = training_arg_parser()
    root_path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../")

    name_params = [
        f"net_name-{args.net_name}_" f"size-{args.img_size}_",
        f'{"multiclass_" if args.multiclass else ""}',
        f'{"tumor_" if args.tumor else ""}',
        f'{"discard_" if args.discard else ""}',
        f"fold-{args.fold}_",
        f"lr-{args.learning_rate}_",
        f"loss-{args.loss_name}",
        f"ww-{args.ww}_",
        f"wl-{args.wl}_",
        f"sched-{args.scheduler}_",
        f"opt-{args.optimizer}_",
        f'{"no_val_" if args.no_val else ""}',
        f"seed-{args.seed}_",
        f"n_layers-{args.n_layers}_" if args.net_name == "transformer" else "",
        f"n_heads-{args.n_heads}_" if args.net_name == "transformer" else "",
        f"d_model-{args.d_model}_" if args.net_name == "transformer" else "",
    ]
    name_params = filter(None, name_params)
    base_name = "_".join(name_params)

    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_dir = os.path.join(root_path, f"{timestamp}_{base_name}")
    if os.path.exists(log_dir):
        print("Log dir already exists")
        exit()
    os.makedirs(log_dir)
    log_file: IO = open(f"{log_dir}/log.log", "w")
    original_out = sys.stdout
    sys.stdout = FileConsoleOut(original_out, log_file)

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

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
        model = create_model(
            args.net_name,
            args.multiclass,
            args.img_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
        ).to(device)
        prepare_polar_images(
            dataset,
            base_name,
            model,
            args.batch_size,
            basic_dir=root_path,
            device=device,
            multiclass=args.multiclass,
        )
        base_name = f"{base_name}polar"

    if args.no_val:
        train, test = dataset.train_val_test_k_fold(0.2, no_val=True, seed=args.seed)
        train_dataset = {
            "train": dataset.create_data_loader(train, args.batch_size, seed=args.seed)
        }
    else:
        folds, test = dataset.train_val_test_k_fold(0.2, seed=args.seed)
        print(folds)

        folds_data_loaders = dataset.create_k_fold_data_loaders(
            folds, batch_size=args.batch_size, seed=args.seed
        )
    test_dataset = dataset.create_data_loader(test, args.batch_size, seed=args.seed)
    print(test)
    rerun = 0
    finished = False
    while not finished:
        name = base_name if rerun == 0 else f"{base_name}_rerun-{rerun}"
        wandb.init(project="master-thesis", entity="s175454", mode="online")
        wandb.config.update(args)
        wandb.run.name = f"{name}-{wandb.run.id}"
        try:
            model = create_model(
                args.net_name,
                args.multiclass,
                args.img_size,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
            ).to(device)
            loss = create_loss(args.loss_name).to(device)
            config = TrainingConfig(
                args.batch_size,
                args.epochs,
                args.learning_rate,
                args.multiclass,
                model,
                loss,
                optim_name=args.optimizer,
                scheduler_name=args.scheduler,
            )
            if args.no_val:
                run_training(name, config, device, train_dataset)
            else:
                run_training(name, config, device, folds_data_loaders[args.fold])
        except RerunException as e:
            rerun += 1
            print(e)
            print("Rerunning")
            wandb.finish()
        except Exception as e:
            print(e)
            raise e
        else:
            finished = True
            test_config = TestingConfig(args.batch_size, args.multiclass, model, loss)
            dataset.test_mode = True
            run_test(name, test_config, device, test_dataset, True)

    sys.stdout = original_out
    log_file.close()


if __name__ == "__main__":
    main()
