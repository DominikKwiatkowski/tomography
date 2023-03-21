import os

import numpy as np
import pandas as pd
import matplotlib.image

metadata_path = "/home/macierz/s175454/lits_prepared/metadata.csv"
dataset = "/home/macierz/s175454/lits_prepared"
# load metadata to numpy array
metadata = pd.read_csv(metadata_path, index_col="id")
metadata = metadata.to_numpy()
metadata = np.delete(metadata, np.where(metadata[:, 5] == 1.0), axis=0)

# patient 43 is mirrored, so we need to mirror it back
patient = 43
patient_data = metadata[np.where(metadata[:, 1] == patient)]
for i in range(len(patient_data)):
    image = np.load(os.path.join(dataset, patient_data[i][3]))["arr_0"]
    label = np.load(os.path.join(dataset, patient_data[i][4]))["arr_0"]
    slice = patient_data[i][0]
    # swap pixels from top and bottom
    image = np.flip(image, 0)
    # save image
    np.savez_compressed(os.path.join(dataset, patient_data[i][3]), image)
