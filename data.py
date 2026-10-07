import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (500, 500)  # (width, height) ปรับตามต้องการ

def get_data(folder_path):
    data_x = []
    if not os.path.isfile(folder_path) and not folder_path.startswith("."):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                x = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                if x is None:          # อ่านไฟล์ไม่ได้
                    continue
                x = cv2.resize(x, IMG_SIZE)
                data_x.append(x.flatten())
        data_x = np.array(data_x)
    return data_x