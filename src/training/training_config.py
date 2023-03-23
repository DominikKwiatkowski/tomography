import copy

from torch.optim.optimizer import Optimizer

import torch
import torch.optim as optim
from timm import optim as timm_optim

from src.optim.scheduler import PolynomialLR


class TrainingConfig:
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        multiclass: bool,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optim_name: str,
        scheduler_name: str,
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
        # self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        if optim_name == "SGD":
            self.optimizer = timm_optim.create_optimizer_v2(
                self.net,
                "sgd",
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0,
            )
        elif optim_name == "Adam":
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise Exception(f"Optimizer {optim_name} not implemented")
        self.scheduler_name = scheduler_name
        if self.scheduler_name == "Polynomial":
            self.scheduler = PolynomialLR(
                self.optimizer,
                step_size=1,
                iter_warmup=0,
                iter_max=self.epochs,
                power=0.9,
                min_lr=1e-5,
            )
        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=self.learning_rate_patience,
                threshold=0.0000001,
                threshold_mode="abs",
            )
        else:
            raise Exception(f"Scheduler {scheduler_name} not implemented")

        self.loss = loss
        self.best_dice = 0.0
        self.best_model_wts = copy.deepcopy(self.net.state_dict())
