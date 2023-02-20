import os
import numpy as np
from scipy.ndimage import convolve


def load_image(
    idx,
    metadata,
    dataset_dir,
    target_size=512,
    transform=None,
    target_transform=None,
    multiclass=False,
    window_width=2560,
    window_center=1536,
    label_map_value=1.0,
):
    image = np.load(os.path.join(dataset_dir, str(metadata[idx][2], encoding="utf-8")))[
        "arr_0"
    ]
    image = np.reshape(image, (1, image.shape[0], image.shape[1]))
    label = np.load(os.path.join(dataset_dir, str(metadata[idx][3], encoding="utf-8")))[
        "arr_0"
    ]
    label = np.reshape(label, (1, label.shape[0], label.shape[1]))

    #
    if multiclass:
        # Duplicate label map to create 3 channel label map
        label = np.repeat(label, 3, axis=0)
        # In first channel, sets 0 to 1, 1 to 0, 2 to 0
        label[0, :, :] = np.where(label[0, :, :] == 0, 1, 0)
        # In second channel, sets 0 to 0, 1 to 1, 2 to 0
        label[1, :, :] = np.where(label[1, :, :] == 1, 1, 0)
        # In third channel, sets 0 to 0, 1 to 0, 2 to 1
        label[2, :, :] = np.where(label[2, :, :] == 2, 1, 0)

    else:
        label = np.where(label >= label_map_value, 1, 0)

    if target_size != image.shape[1]:
        factor = int(image.shape[1] / target_size)
        filter = np.ones((1, factor, factor)) / (factor ** 2)
        # reshape all images and labels to target size using downscaling
        image = convolve(image, filter)[:, 0::factor, 0::factor]
        label_sampled = convolve(label, filter)[:, 0::factor, 0::factor]
        # check if label valuesare binary, if not, make them binary
        label = np.where(label_sampled > 0.5, 1, 0)

    # Apply winow width and center
    image = np.clip(
        image,
        window_center - window_width,
        window_center + window_width,
    )
    image = (image - (window_center - window_width)) / window_width - 1
    if transform:
        image = transform(image)
    if target_transform:
        label = target_transform(label)

    return image, label
