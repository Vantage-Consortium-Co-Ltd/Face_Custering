from data import get_data 
from calculate import custering_calculate 

def main():
    folder_path = "Dataset"
    data_x = get_data(folder_path)
    print("Data shape:", data_x.shape)
    #plot_data(data_x)
    result = custering_calculate(data_x, k=3)
    print("Clustering result:", result)


if __name__ == "__main__":
    main()