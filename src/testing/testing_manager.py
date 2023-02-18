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
    utils.load_model(testing_config.net, weights_filename, device)

    if torch.cuda.is_available():
        testing_config.net.cuda(device)

    utils.run_test_epoch(testing_config, data_loader, device)
