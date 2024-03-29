import copy
import os
from collections import defaultdict
from typing import Tuple, Dict

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader
from torchmetrics import Dice
from tqdm import tqdm

from src.models.DefEDNet import DefED_Net
from src.models.NestedUnet import UNet, NestedUNet
from src.models.QAU_Net import QAU_Net
from src.polar_transforms import to_cart
from src.testing.testing_config import TestingConfig
from src.training.training_config import TrainingConfig
from src.models.transformer.factory import create_segmenter


class TestingImages:
    def __init__(self):
        self.best_dice = -1.0
        self.best_dice_img = None
        self.best_dice_mask = None
        self.best_dice_pred = None

        self.worst_dice = 1.1
        self.worst_dice_img = None
        self.worst_dice_mask = None
        self.worst_dice_pred = None


def multi_label_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    num_of_classes: int,
) -> float:
    """
    Calculate the dice score for a batch of images.
    :param preds: Batch of predicted masks
    :param targets: Batch of target masks
    :param device: device str
    :param num_of_classes: number of classes
    :return: Dice score, mean of all classes
    """
    dice = Dice(multiclass=True, num_classes=num_of_classes, ignore_index=0).to(device)
    return dice(preds, targets).item()


def calc_dice_and_check(
    pred: torch.Tensor,
    target: torch.Tensor,
    input: torch.Tensor,
    imageStats: TestingImages,
    device: torch.device,
    num_of_classes: int,
) -> TestingImages:
    """
    Calculate dice score for each image, if it is better/worse than previous best/worst,
    update the image.
    """
    dice_coeff_metric = Dice().to(device)
    for i in range(pred.shape[0]):
        if num_of_classes == 1:
            dice_score = dice_coeff_metric(pred[i], target[i]).item()
        else:
            dice_liver = dice_coeff_metric(pred[i, 1], target[i, 1]).item()
            dice_tumor = dice_coeff_metric(pred[i, 2], target[i, 2]).item()
            dice_score = (dice_liver + dice_tumor) / 2
        if dice_score > imageStats.best_dice:
            imageStats.best_dice = dice_score
            imageStats.best_dice_img = input[i]
            imageStats.best_dice_mask = target[i]
            imageStats.best_dice_pred = pred[i]
        if dice_score < imageStats.worst_dice:
            imageStats.worst_dice = dice_score
            imageStats.worst_dice_img = input[i]
            imageStats.worst_dice_mask = target[i]
            imageStats.worst_dice_pred = pred[i]
    return imageStats


def calc_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: dict,
    device: torch.device,
    num_of_classes: int,
) -> None:
    dice_coeff_metric = Dice().to(device)
    if num_of_classes == 1:
        dice_score = dice_coeff_metric(pred, target)
        metrics["dice"] += dice_score.item() * target.size(0)
    else:
        dice_liver = dice_coeff_metric(pred[:, 1], target[:, 1])
        dice_tumor = dice_coeff_metric(pred[:, 2], target[:, 2])
        dice_score = (dice_liver + dice_tumor) / 2

        metrics["dice_liver"] += dice_liver.item() * target.size(0)
        metrics["dice_tumor"] += dice_tumor.item() * target.size(0)
        metrics["dice"] += dice_score.item() * target.size(0)


def print_metrics(metrics: dict, epoch_samples: int, phase: str) -> None:
    """
    Print out the metrics to stdout.
    :param tb: SummaryWriter object to write to
    :param metrics: dictionary containing the metrics
    :param epoch_samples: number of samples in this epoch
    :param phase: train/val
    :param epoch: current epoch
    """
    outputs = []
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
        outputs.append("{}: {:4f}".format(k, metrics[k]))
    print("{}: {}".format(phase, ", ".join(outputs)))
    # to each key add phase
    metrics = {f"{phase}_{k}": v for k, v in metrics.items()}
    wandb.log(metrics)


def save_model(net: nn.Module, name: str) -> None:
    """
    Save the model to disk.
    """
    torch.save(net.state_dict(), os.path.join(wandb.run.dir, name))


def load_model(
    net: nn.Module, name: str, device: torch.device, during_training: bool = False
) -> None:
    """
    Load the model from disk.
    """
    if during_training:
        net.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, name), map_location=device)
        )
    else:
        net.load_state_dict(torch.load(name, map_location=device))


class BinaryDiceLoss(nn.Module):
    r"""Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=1, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.mul(torch.sum(torch.mul(predict, target), dim=1), 2) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception(f"Unexpected reduction {self.reduction}")


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape do not match"
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert (
                        self.weight.shape[0] == target.shape[1]
                    ), "Expect weight shape [{}], get[{}]".format(
                        target.shape[1], self.weight.shape[0]
                    )
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / (target.shape[1] - (0 if self.ignore_index is None else 1))


def create_model(
    net_name: str = "",
    multiclass: bool = True,
    img_size: int = 512,
    n_layers: int = 12,
    n_heads: int = 12,
    d_model: int = 768,
) -> nn.Module:
    if net_name == "unet":
        return UNet(
            input_channels=1,
            num_classes=3 if multiclass else 1,
        ).float()
    elif net_name == "quanet":
        return QAU_Net(
            in_ch=1,
            out_ch=3 if multiclass else 1,
        ).float()
    elif net_name == "defednet":
        return DefED_Net(
            num_classes=3 if multiclass else 1,
        ).float()
    elif net_name == "unetplusplus":
        return NestedUNet(
            input_channels=1,
            num_classes=3 if multiclass else 1,
        ).float()
    elif net_name == "transformer":
        if d_model == 192:
            backbone = "vit_tiny_patch16_384"
        elif d_model == 384:
            backbone = "vit_small_patch16_384"
        elif d_model == 768:
            backbone = "vit_base_patch16_384"
        elif d_model == 1024:
            backbone = "vit_large_patch16_384"
        else:
            raise ValueError("Unknown d_model")
        cfg = dict(
            backbone=backbone,
            d_model=d_model,
            decoder=dict(
                drop_path_rate=0.1,
                dropout=0.1,
                n_cls=3 if multiclass else 1,
                n_layers=2,
                name="mask_transformer",
            ),
            distilled=False,
            drop_path_rate=0.1,
            dropout=0.0,
            image_size=(img_size, img_size),
            n_cls=3 if multiclass else 1,
            n_layers=n_layers,
            n_heads=n_heads,
            patch_size=16,
            channels=1,
        )
        return create_segmenter(cfg)

    else:
        raise ValueError("Unknown net name")


def norm_point(point):
    return (point + 1024) / 2560 - 1


def get_class_labels() -> Dict[int, str]:
    if wandb.config["multiclass"]:
        class_labels = {
            0: "Background",
            1: "Liver",
            2: "Tumor",
        }
    elif wandb.config["tumor"]:
        class_labels = {
            0: "Background",
            1: "Tumor",
        }
    else:
        class_labels = {
            0: "Background",
            1: "Liver",
        }
    return class_labels


def log_image_to_wandb(pred, inputs, labels, name):
    output_tres = torch.where(
        pred < 0.5,
        torch.zeros_like(pred),
        torch.ones_like(pred),
    )
    class_labels = get_class_labels()
    if wandb.config["multiclass"]:
        output_tres = torch.argmax(output_tres, dim=0)
        labels = torch.argmax(labels, dim=0)
    else:
        # if not multiclass, remove second dimension
        output_tres = output_tres.squeeze(0)
        labels = labels.squeeze(0)
    image = wandb.Image(
        inputs.to("cpu").numpy(),
        masks={
            "predictions": {
                "mask_data": output_tres.to("cpu").numpy(),
                "class_labels": class_labels,
            },
            "ground_truth": {
                "mask_data": labels.to("cpu").numpy(),
                "class_labels": class_labels,
            },
        },
    )
    wandb.log({name: image})


def log_worst_and_best_images(
    imageStats: TestingImages,
) -> None:
    log_image_to_wandb(
        imageStats.best_dice_pred,
        imageStats.best_dice_img,
        imageStats.best_dice_mask,
        "Best Dice",
    )
    log_image_to_wandb(
        imageStats.worst_dice_pred,
        imageStats.worst_dice_img,
        imageStats.worst_dice_mask,
        "Worst Dice",
    )
    wandb.log({"worst Dice": imageStats.worst_dice})
    wandb.log({"best Dice": imageStats.best_dice})


def log_image(inputs, labels, outputs):
    image_num = 2 if inputs.shape[0] > 2 else 0
    # If output value less than 0.5, set to 0, else set to 1
    log_image_to_wandb(
        outputs[image_num], inputs[image_num], labels[image_num], "Predictions"
    )


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lossDice = DiceLoss()
        self.lossEntropy = nn.BCELoss()

    def forward(self, predict, target):
        return self.lossDice(predict, target) + self.lossEntropy(predict, target)


def create_loss(name: str) -> nn.Module:
    if name == "dice":
        return DiceLoss()
    elif name == "cross_entropy":
        return nn.BCELoss()
    elif name == "mix":
        return MixLoss()
    else:
        raise ValueError("Unknown loss name")


def run_training_epoch(
    training_config: TrainingConfig,
    training_data_loader: DataLoader,
    epoch: int,
    device: torch.device,
) -> dict:
    training_config.net.train()
    metrics: dict = defaultdict(float)
    epoch_samples = 0
    with tqdm(
        total=len(training_data_loader) * training_config.batch_size,
        desc=f"Epoch {epoch}/{training_config.epochs}",
        unit="img",
    ) as pbar:
        for (inputs, labels, _, _) in training_data_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            training_config.optimizer.zero_grad()
            outputs = training_config.net(inputs)
            loss = training_config.loss(outputs, labels)
            metrics["loss"] += loss.item() * inputs.size(0)
            calc_metrics(
                outputs,
                labels.to(device, dtype=torch.long),
                metrics,
                device,
                training_config.classes,
            )
            loss.backward()
            training_config.optimizer.step()
            epoch_samples += inputs.size(0)
            pbar.update(inputs.size(0))
            pbar.set_postfix(**{"loss (batch)": loss.item()})
    print_metrics(metrics, epoch_samples, "Train")
    # if no validation and polynomial LR, update LR
    if wandb.config["scheduler"] == "Polynomial" and wandb.config["no_val"]:
        training_config.scheduler.step()
    return metrics


def detransform_image(inputs, center_x, center_y):
    inputs = inputs.cpu().detach().numpy()
    for i in range(inputs.shape[0]):
        inputs[i] = to_cart(inputs[i], (center_x[i].item(), center_y[i].item()))
    return torch.from_numpy(inputs)


def eval_run(
    net: nn.Module,
    loss: nn.Module,
    eval_data_loader: DataLoader,
    epoch: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    classes: int,
    phase: str,
) -> Tuple[dict, int]:
    net.eval()
    metrics: dict = defaultdict(float)
    epoch_samples = 0
    image_stats = TestingImages()
    with torch.no_grad():
        with tqdm(
            total=len(eval_data_loader) * batch_size,
            desc=f"Epoch {epoch}/{epochs}",
            unit="img",
        ) as pbar:
            for (inputs, labels, center_x, center_y) in eval_data_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                outputs = net(inputs)
                if phase == "test" and wandb.config["use_polar"]:
                    outputs = outputs.cpu().detach().numpy()
                    for i in range(outputs.shape[0]):
                        outputs[i] = to_cart(
                            outputs[i], (center_x[i].item(), center_y[i].item())
                        )
                    outputs = torch.from_numpy(outputs).to(device, dtype=torch.float32)
                loss_value = loss(outputs, labels)
                metrics["loss"] += loss_value.item() * inputs.size(0)
                calc_metrics(
                    outputs,
                    labels.to(device, dtype=torch.long),
                    metrics,
                    device,
                    classes,
                )
                if phase == "test":
                    if wandb.config["use_polar"]:
                        inputs = detransform_image(inputs, center_x, center_y)
                    image_stats = calc_dice_and_check(
                        outputs,
                        labels.to(device, dtype=torch.long),
                        inputs,
                        image_stats,
                        device,
                        classes,
                    )
                    if epoch_samples == 0:
                        log_image(inputs, labels, outputs)
                epoch_samples += inputs.size(0)
                pbar.update(inputs.size(0))
                pbar.set_postfix(**{"loss (batch)": loss_value.item()})
        if phase == "test":
            log_worst_and_best_images(image_stats)
        return metrics, epoch_samples


def run_val_epoch(
    training_config: TrainingConfig,
    val_data_loader: DataLoader,
    epoch: int,
    device: torch.device,
) -> dict:
    metrics, epoch_samples = eval_run(
        training_config.net,
        training_config.loss,
        val_data_loader,
        epoch,
        device,
        training_config.batch_size,
        training_config.epochs,
        training_config.classes,
        "val",
    )
    print_metrics(metrics, epoch_samples, "Val")
    if training_config.scheduler_name == "Polynomial":
        training_config.scheduler.step()
    else:
        training_config.scheduler.step(metrics["loss"])
    if metrics["dice"] > training_config.best_dice:
        training_config.best_dice = metrics["dice"]
        training_config.best_model_wts = copy.deepcopy(training_config.net.state_dict())
        print(f"Best model saved with dice: {training_config.best_dice}")
    return metrics


def run_test_epoch(
    testing_config: TestingConfig, test_data_loader: DataLoader, device: torch.device
) -> dict:
    metrics, epoch_samples = eval_run(
        testing_config.net,
        testing_config.loss,
        test_data_loader,
        0,
        device,
        testing_config.batch_size,
        1,
        testing_config.classes,
        "test",
    )
    print_metrics(metrics, epoch_samples, "Test")
    return metrics
