import os
import cv2
import numpy as np

IMG_SIZE = (240, 300)

def get_data(folder_path):
    data_x = []
    file_names = []
    if not os.path.exists(folder_path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_path)
        if os.path.exists(alt_path):
            folder_path = alt_path

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                x = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                if x is None:          # อ่านไฟล์ไม่ได้
                    continue
                x = cv2.resize(x, IMG_SIZE)
                data_x.append(x.flatten())
                file_names.append(file)
        data_x = np.array(data_x)
    return data_x, file_names