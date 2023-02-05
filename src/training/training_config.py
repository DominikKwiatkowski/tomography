from torch.optim.optimizer import Optimizer

import torch
import torch.optim as optim


class TrainingConfig:
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        multiclass: bool,
        net: torch.nn.Module,
    ):
        # Batch size for training
        self.batch_size: int = batch_size

        # Number of folds for cross validation
        self.k_folds: int = 5

        # Number of epochs to train for
        self.epochs: int = epochs

        # Learning rate
        self.learning_rate: float = learning_rate
        self.learning_rate_patience: int = 3

        # Input shape
        self.input_h = 512
        self.input_w = 512
        self.channels = 1
        # background, liver, tumor
        self.classes = 3 if multiclass else 1
        # Mode layers definition
        self.net = net
        print(self.net)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.optimizer: Optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=self.learning_rate_patience,
            threshold=0.0000001,
            threshold_mode="abs",
        )
