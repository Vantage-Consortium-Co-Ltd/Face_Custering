import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(data_x, labels, title="Cluster Plot"):
    """
    Project image vectors to 2D using PCA and draw a scatter plot
    with one colored point per cluster, marking centroids as 'X'.

    Parameters:
        data_x (np.ndarray): Image vectors of shape (N, D).
        labels (np.ndarray): Cluster assignments of shape (N,).
        title (str): Title of the plot.
    """
    if len(data_x) == 0:
        print("Warning: data_x is empty, skipping plot.")
        return

    # Center the data
    data_centered = data_x.astype(np.float64) - np.mean(data_x, axis=0)

    # 2D PCA using SVD (efficient for high-dimensional data where N << D)
    u, s, _ = np.linalg.svd(data_centered, full_matrices=False)
    coords_2d = u[:, :2] * s[:2]

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    # Generate distinct colors for each cluster
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        points = coords_2d[mask]
        color = colors[i]

        # Draw data points for this cluster
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            label=f"Cluster {label}",
            alpha=0.8,
            edgecolors="none",
            s=60
        )

        # Calculate and plot cluster centroid as 'X'
        if len(points) > 0:
            centroid = np.mean(points, axis=0)
            plt.scatter(
                centroid[0],
                centroid[1],
                marker="x",
                s=160,
                color="black",
                linewidths=2.5,
                label=f"Centroid {label}" if i == 0 else ""  # avoid cluttered legend
            )

    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1 (PC 1)", fontsize=11)
    plt.ylabel("Principal Component 2 (PC 2)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()