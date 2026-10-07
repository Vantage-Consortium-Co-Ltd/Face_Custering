# Face_Custering

Face image clustering using K-means implemented from scratch with NumPy.

## How it works

1. `get_data` reads every `.jpg` in the dataset folder as grayscale, resizes it to `IMG_SIZE` (default 500x500) and flattens it into one vector per image.
2. `custering_calculate` runs K-means on those vectors (mean squared distance, random initial centroids) and returns the cluster index of each image.
3. `plot_data` projects the image vectors to 2D with PCA and draws one point per image.

## Project structure

| File | Description |
|---|---|
| `main.py` | Entry point: load data, run clustering, print result |
| `data.py` | `get_data` (load and preprocess images), `plot_data` (2D plot) |
| `calculate.py` | `custering_calculate` (K-means) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Put the face images (`.jpg`) in a `Dataset/` folder next to `main.py`. The folder is git-ignored.

## Usage

```bash
python main.py
```

Change the number of clusters with `k` in `main.py`, and the image size with `IMG_SIZE` in `data.py`.

## Notes

- All images are resized to the same size, otherwise the flattened vectors have different lengths and `np.array` fails.
- K-means starts from random centroids, so results can differ between runs. Empty clusters keep their previous centroid.
