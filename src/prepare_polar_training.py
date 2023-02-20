import numpy as np
import torch

import pathlib

from src.data_loader import TomographyDataset
from src.polar_transforms import centroid
from tqdm import tqdm


def add_centroid_to_metadata(
    dataset: TomographyDataset,
    net: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    lcc: bool = False,
    use_ground_truth: bool = False,
    multiclass: bool = False,
) -> None:
    # for each image in metadata, add centroid. Load image, find centroid, add to metadata,
    # if there is a 100% background image, add (0,0) to metadata
    # return metadata
    metadata = dataset.metadata
    # add two columns to metadata
    metadata = np.append(metadata, np.zeros((len(metadata), 2)), axis=1)
    label_index = 1 if multiclass else 0
    if use_ground_truth:
        for i in tqdm(range(len(metadata))):
            label = dataset[i][1].astype(dtype=np.float32)
            center = centroid(label[label_index], lcc)
            metadata[i, 7] = center[0]
            metadata[i, 8] = center[1]
    else:
        with torch.no_grad():
            for i in tqdm(range(int(len(metadata) / batch_size))):
                batch = []
                for j in range(batch_size):
                    if i * batch_size + j < len(metadata):
                        img = dataset[i * batch_size + j][0]
                        batch.append(img)
                    else:
                        batch.append(np.zeros((1, 512, 512)))
                # move to numpy
                batch = np.array(batch)
                batch_torch = torch.from_numpy(batch)
                batch_torch = batch_torch.to(device, dtype=torch.float)
                output = net(batch_torch)
                output = output.cpu().detach().numpy()
                for j in range(len(output)):
                    index = i * batch_size + j
                    if index >= len(metadata):
                        break
                    img = output[j]
                    # threshold image
                    img[img > 0.5] = 1
                    img[img <= 0.5] = 0
                    center = centroid(img[label_index], lcc)
                    metadata[index, 7] = center[0]
                    metadata[index, 8] = center[1]
    dataset.metadata = metadata
    dataset.polar = True


def find_net(net_name, basic_dir):
    """
    Find the network that was trained on the dataset recursive. If the network is not found, return None.
    If more than one network is found, print all the networks and return None.
    """
    nets = []
    root = pathlib.Path(basic_dir)
    for path in root.rglob(net_name):
        nets.append(path)

    if len(nets) == 0:
        return None
    elif len(nets) == 1:
        return nets[0]
    else:
        print("More than one network found. Choose one of the following networks:")
        for i in range(len(nets)):
            print(i, nets[i])
        return None


def prepare_polar_images(
    dataset: TomographyDataset,
    name: str,
    model: torch.nn.Module,
    batch_size: int,
    lcc: bool = False,
    basic_dir: str = "",
    device: torch.device = None,
    multiclass: bool = False,
) -> None:
    net_path = find_net(name, basic_dir)
    if net_path is None:
        print("Network not found")
        return
    model.load_state_dict(torch.load(net_path, map_location=device))
    model.to(device)
    model.eval()
    add_centroid_to_metadata(
        dataset, model, batch_size, device, lcc, multiclass=multiclass
    )
