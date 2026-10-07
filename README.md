# Face_Custering

Face image clustering implemented from scratch with NumPy. Two algorithms are provided: hard K-means and soft (fuzzy) mean.

## How it works

1. `get_data` reads every `.jpg` in the dataset folder as grayscale, resizes it to `IMG_SIZE` (default 240x300) and flattens it into one vector per image. It returns the vectors together with the file names, in the same order.
2. `custering_calculate` runs K-means on those vectors (mean squared distance, random initial centroids) and returns the cluster index of each image.
3. `soft_mean_calculate` runs soft-mean (fuzzy c-means) with fuzzifier `m = 1.5`, prints the centroid change on every iteration until it converges, and returns the cluster with the highest membership for each image.
4. `plot_clusters` projects the image vectors to 2D with PCA and draws one colored point per cluster, with the centroids marked as X.

## Project structure

```
Face_Custering/
├── Dataset/                 # face images (git-ignored, create it yourself)
│   ├── example.jpg
│   ├── example_2.jpg
│   └── ...
├── main.py                  # entry point
├── data.py                  # image loading and preprocessing
├── calculate.py             # clustering algorithms
├── requirements.txt         # Python dependencies
└── README.md
```

| File | Description |
|---|---|
| `main.py` | Entry point: load data, run both clustering methods, print `Name: <file>, Cluster: <n>` and plot each result |
| `data.py` | `get_data` (load and preprocess images, returns `data_x, file_names`) |
| `calculate.py` | `custering_calculate` (K-means), `soft_mean_calculate` (soft mean) |
| `plot.py` | `plot_clusters` (PCA 2D scatter plot of a clustering result) |
| `requirements.txt` | Python dependencies |

## Dataset layout

Put the face images directly inside `Dataset/`, next to `main.py`. Only `.jpg` files are read, and subfolders are not scanned.

```
Dataset/
├── example.jpg
├── example_2.jpg
└── example_3.jpg
```

The file name is what gets printed next to the cluster number, so name each file after the person or image you want to recognize in the output, for example `example.jpg`. The folder is git-ignored and is not included in the repository.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Example output:

```
Data shape: (18, 72000)
Hard Mean Clustering Results:
Name: example.jpg, Cluster: 0
Name: example_2.jpg, Cluster: 2
...
```

Change the number of clusters with `k` in `main.py`, and the image size with `IMG_SIZE` in `data.py`.

## Notes

- All images are resized to the same size, otherwise the flattened vectors have different lengths and `np.array` fails.
- Both algorithms start from random values, so cluster numbers (and sometimes the grouping) can differ between runs.
- K-means: an empty cluster keeps its previous centroid.
- Soft mean: with a large fuzzifier (`m = 2`) on raw pixels, all centroids can collapse to the overall mean and one cluster ends up empty. That is why `m` is set to 1.5.
- Raw pixels mostly capture lighting and background rather than identity, so clusters may not match people.
- `.heif` and other non-`.jpg` files in `Dataset/` are skipped.
