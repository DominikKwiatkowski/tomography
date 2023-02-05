import torch


class TestingConfig:
    def __init__(
        self,
        batch_size: int,
        multiclass: bool,
        net: torch.nn.Module,
    ):
        # Batch size for training
        self.batch_size: int = batch_size

        # Input shape
        self.input_h = 512
        self.input_w = 512
        self.channels = 1
        # background, liver, tumor
        self.classes = 3 if multiclass else 1
        # Mode layers definition
        self.net = net
