from datetime import datetime

import wandb
from torch import optim, nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.training.training_config import TrainingConfig
import copy
import torch
import time
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import gc


class RerunException(Exception):
    pass


def run_training(
    training_name: str,
    training_config: TrainingConfig,
    device: torch.device,
    data_loaders: dict,
) -> None:
    """
    Runs the training loop.
    :param device:
    :param training_config:
    :param training_name: name of the training
    :param data_loaders: dictionary of dataloaders
    """
    print(f"Training {training_name} on device: {device}")

    if torch.cuda.is_available():
        training_config.net.cuda(device)
    print(sum(p.numel() for p in training_config.net.parameters() if p.requires_grad))

    try:
        for epoch in range(training_config.epochs):
            print("Epoch {}/{}".format(epoch, training_config.epochs - 1))
            print("-" * 10)

            since = time.time()
            # Each epoch has a training and validation phase
            for param_group in training_config.optimizer.param_groups:
                print("LR", param_group["lr"])
            utils.run_training_epoch(
                training_config, data_loaders["train"], epoch, device
            )
            if not wandb.config["no_val"]:
                metrics = utils.run_val_epoch(
                    training_config, data_loaders["val"], epoch, device
                )
                if wandb.config["early_stop"] and epoch >= 20 and metrics["dice"] < 0.5:
                    raise RerunException(f"Dice under 50 in epoch {epoch}, rerunning")

            time_elapsed = time.time() - since
            print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            gc.collect()
            torch.cuda.empty_cache()

    finally:
        print(f"Best val dice: {training_config.best_dice:4f}")
        # If we are not using validation, we will save the last model
        if not wandb.config["no_val"]:
            # load best model weights
            training_config.net.load_state_dict(training_config.best_model_wts)
        # save model
        utils.save_model(training_config.net, training_name)
