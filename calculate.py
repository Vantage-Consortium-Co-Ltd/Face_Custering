import numpy as np

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
        if np.mean(np.abs(c_old - centroids)) < 1e-6:
            break
    
    return idx

def soft_mean_calculate(x, k, m=1.5, max_iter=100, tol=1e-2):
    x = x.astype(np.float64)          # uint8 จะ overflow ตอนลบกัน และเก็บ centroid เป็นทศนิยมไม่ได้
    centroids = x[np.random.permutation(len(x))[:k]]
    mu = np.random.rand(len(x), k)
    for _ in range(max_iter):
        mu /= np.sum(mu, axis=1)[:, None]
        c_old = centroids.copy()
        for i in  range(k):
            centroids[i] = np.dot(mu[:,  i] ** m,  x) / np.sum(mu[:, i] ** m)
            distance = np.sqrt(np.sum((x - centroids[i]) ** 2, axis=1))
            distance = np.maximum(distance, 1e-12)   # กันหารด้วย 0 เมื่อจุดทับ centroid
            distance = distance ** (-2 / (m - 1))
            mu[:, i] = distance
        change = np.mean(np.abs(c_old - centroids))
        if change < tol:
            break

    return np.argmax(mu, axis=1)
