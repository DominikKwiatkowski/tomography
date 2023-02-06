import os

import torch
import torch.nn as nn
import wandb
from torchmetrics import Dice

from src.models.DefEDNet import DefED_Net
from src.models.NestedUnet import UNet, NestedUNet
from src.models.QAU_Net import QAU_Net


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


def print_metrics(
        metrics: dict, epoch_samples: int, phase: str
) -> None:
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


def load_model(net: nn.Module, name: str) -> None:
    """
    Load the model from disk.
    """
    net.load_state_dict(torch.load(os.path.join(wandb.run.dir, name)))

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

    def __init__(self, smooth=1, p=2, reduction="mean"):
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

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
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


def create_model(net_name: str = "", multiclass: bool = True) -> nn.Module:
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
    else:
        raise ValueError("Unknown net name")


def norm_point(point):
    return (point + 1024) / 2560 - 1


def log_image(inputs, labels, outputs):
    image_num = 2 if inputs.shape[0] > 2 else 0
    # If output value less than 0.5, set to 0, else set to 1
    output_tres = torch.where(
        outputs[image_num] < 0.5,
        torch.zeros_like(outputs[0]),
        torch.ones_like(outputs[0]),
    )
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

    # If multiclass, convert 3 channels to 1 channel, such as if first channel is 1, set to 0, if second channel is 1, set to 1, if third channel is 1, set to 2
    if wandb.config["multiclass"]:
        output_tres = torch.argmax(output_tres, dim=0)
        labels = torch.argmax(labels, dim=1)


    # Log image with wandb
    image = wandb.Image(inputs[image_num].to("cpu").numpy(), masks={
        "predictions": {
            "mask_data": output_tres.to("cpu").numpy(),
            "class_labels": class_labels,
        },
        "ground_truth": {
            "mask_data": labels[image_num].to("cpu").numpy(),
            "class_labels": class_labels,
        }})
    wandb.log({"predictions": image})