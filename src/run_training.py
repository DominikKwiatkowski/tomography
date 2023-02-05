import argparse
import wandb

from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.testing.testing_config import TestingConfig
from src.testing.testing_manager import run_test
from src.training.training_config import TrainingConfig
from training.training_manager import run_training, RerunException
from src.utils import create_model
import torch
import torch.multiprocessing as mp


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
    parser.add_argument("--net_name", type=str, help="Network name", choices=["unet", "quanet", "defednet", "unetplusplus"], required=True)
    parser.add_argument("--tumor", action="store_true", help="Use tumor labels")
    parser.add_argument("--normalize", action="store_true", help="Normalize images")
    parser.add_argument(
        "--discard", action="store_true", help="Discard images with 100% background"
    )
    parser.add_argument("--multiclass", action="store_true", help="Use multiclass")
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = training_arg_parser()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    name_params = [
        f"net_name-{args.net_name}"
        f"size-{args.img_size}",
        f'{"multiclass" if args.multiclass else ""}',
        f'{"tumor" if args.tumor else ""}',
        f'{"normalize" if args.normalize else ""}',
        f'{"discard" if args.discard else ""}',
        f"fold-{args.fold}",
        f"lr-{args.learning_rate}",
    ]

    metadata = load_metadata(args.metadata)

    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(
        args.dataset,
        metadata,
        target_size=args.img_size,
        tumor=args.tumor,
        normalize=args.normalize,
        discard=args.discard,
        multiclass=args.multiclass,
    )

    folds, test = dataset.train_val_test_k_fold(0.2)
    print(folds)

    folds_data_loaders = dataset.create_k_fold_data_loaders(
        folds, batch_size=args.batch_size
    )
    test_dataset = dataset.create_data_loader(test, args.batch_size)

    name_params = filter(None, name_params)
    base_name = "_".join(name_params)
    rerun = 0
    finished = False
    while not finished:
        name = base_name if rerun == 0 else f"{base_name}_rerun-{rerun}"
        wandb.init(project="master-thesis", entity="s175454", mode="online")
        wandb.config.update(args)
        wandb.run.name = f"{name}-{wandb.run.id}"
        try:
            model = create_model(args.net_name, args.multiclass).to(device)
            config = TrainingConfig(
                args.batch_size,
                args.epochs,
                args.learning_rate,
                args.multiclass,
                model
            )
            run_training(
                name,
                config,
                device,
                folds_data_loaders[args.fold],
            )
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
            test_config = TestingConfig(args.batch_size, args.multiclass, model)
            run_test(name, test_config, device, test_dataset)


if __name__ == "__main__":
    main()
