from torch.utils.data import DataLoader

from datetime import datetime
import torch
import src.utils as utils
from collections import defaultdict
from tqdm import tqdm
import gc

from src.testing.testing_config import TestingConfig


def run_test(
    weights_filename: str,
    testing_config: TestingConfig,
    device: torch.device,
    data_loader: DataLoader,
) -> None:
    """
    Runs the training loop.
    :param device:
    :param testing_config:
    :param weights_filename: name of the weights file
    :param data_loader: DataLoader object
    """
    print(f"Testing {weights_filename} on device: {device}")
    utils.load_model(testing_config.net, weights_filename)

    if torch.cuda.is_available():
        testing_config.net.cuda(device)

    with torch.no_grad():
        testing_config.net.eval()  # Set model to evaluate mode
        metrics: dict = defaultdict(float)
        test_samples = 0
        with tqdm(
            total=len(data_loader) * testing_config.batch_size,
            desc=f"Testing {weights_filename}",
            unit="img",
        ) as pbar:
            for (inputs, labels) in data_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)

                outputs = testing_config.net(inputs)
                # if it is first sample, save first image to tensorboard

                loss_value = testing_config.loss(outputs, labels)
                metrics["loss"] += loss_value.item() * inputs.size(0)
                if test_samples == 0:
                    utils.log_image(inputs,labels, outputs)
                utils.calc_metrics(
                    outputs, labels.to(device, dtype=torch.long), metrics, device, testing_config.classes
                )

                test_samples += inputs.size(0)
                pbar.update(inputs.size(0))
                pbar.set_postfix(**{"loss (batch)": loss_value.item()})

        utils.print_metrics(metrics, test_samples, "test")
