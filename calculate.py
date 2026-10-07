import cv2
import numpy as np
import matplotlib.pyplot as plt

def custering_calculate(x, k):
    x = x.astype(np.float64)
    centroids = x[np.random.permutation(len(x))[:k]]
    while True:
        D = np.zeros((len(x), k))
        for i in range(k):
            D[:, i] = np.mean((x - centroids[i])**2, axis=1)
        idx = np.argmin(D, axis=1)

        c_old = centroids.copy()
        for i in range(k):
            if np.any(idx == i):          # กัน cluster ว่าง
                centroids[i] = np.mean(x[idx == i], axis=0)
        if np.mean(np.abs(c_old - centroids)) == 0:
            break
    return idx
