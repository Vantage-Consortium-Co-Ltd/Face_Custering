from data import get_data 
from calculate import custering_calculate, soft_mean_calculate
from plot import plot_clusters

def main():
    folder_path = "Dataset"
    data_x, file_names = get_data(folder_path)
    print("Data shape:", data_x.shape)

    result_hardmean = custering_calculate(data_x, k=3)
    print("Hard Mean Clustering Results:")
    for name, label in zip(file_names, result_hardmean):
        print(f"Name: {name}, Cluster: {label}")
    plot_clusters(data_x, result_hardmean)

    result_softmean = soft_mean_calculate(data_x, k=3)
    print("\nSoft Mean Clustering Results:")
    for name, label in zip(file_names, result_softmean):
        print(f"Name: {name}, Cluster: {label}")
    plot_clusters(data_x, result_softmean)
    


if __name__ == "__main__":
    main()