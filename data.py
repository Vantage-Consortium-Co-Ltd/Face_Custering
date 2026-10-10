import os
import numpy as np
import cv2

IMG_SIZE = (240, 300)

def get_data(folder_path):
    data_x = []
    file_names = []
    if not os.path.isfile(folder_path) and not folder_path.startswith("."):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                x = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                if x is None:          
                    continue
                x = cv2.resize(x, IMG_SIZE)
                data_x.append(x.flatten())
                file_names.append(file)
        data_x = np.array(data_x)
    return data_x, file_names