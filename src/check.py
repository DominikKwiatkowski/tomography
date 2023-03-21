import os

import numpy as np
import pandas as pd
import matplotlib.image

metadata_path = "/home/macierz/s175454/lits_prepared/metadata.csv"
dataset = "/home/macierz/s175454/lits_prepared"
save = "/home/macierz/s175454/lits-photo"
# load metadata to numpy array
metadata = pd.read_csv(metadata_path, index_col="id")
metadata = metadata.to_numpy()
metadata = np.delete(metadata, np.where(metadata[:, 5] == 1.0), axis=0)

# print largest liver slice for each patient, slice num is 0, patient num is 1
for patient in range(131):
    patient_data = metadata[np.where(metadata[:, 1] == patient)]
    largest_liver_slice = patient_data[np.argmax(patient_data[:, 6])]
    image = np.load(os.path.join(dataset, largest_liver_slice[3]))["arr_0"]
    label = np.load(os.path.join(dataset, largest_liver_slice[4]))["arr_0"]
    slice = largest_liver_slice[0]
    # save image
    matplotlib.image.imsave(
        os.path.join(save, f"patient_{patient}_slice_{slice}.png"), image, cmap="gray"
    )
    matplotlib.image.imsave(
        os.path.join(save, f"patient_{patient}_slice_{slice}_label.png"), label
    )
