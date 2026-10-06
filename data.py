import os
import cv2
import numpy as np
def get_data(folder_path):
    data_x = []
    if not os.path.isfile(folder_path) and not folder_path.startswith("."):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                x = cv2.imread(folder_path + "/" + file, cv2.IMREAD_GRAYSCALE)
                data_x.append(x.flatten()) 
        data_x = np.array(data_x)
    return data_x