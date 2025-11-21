import os
import numpy as np
from tresnet.dataset import CustomDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dtaidistance import dtw
from dtaidistance.dtw import distance_matrix, distance_matrix_fast
from sklearn.neighbors import KNeighborsClassifier
import pdb 
from tqdm import tqdm
from soft_dtw_cuda import SoftDTW
import torch 

def calculate_sdtw_matrix(x, y):
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    # x and y are assumed to be in (batch, seq_len, dim)
    x_n_samples = x.shape[0]
    y_n_samples = y.shape[0]
    train_distance_matrix = np.zeros((x_n_samples, y_n_samples))
    yy_sdtw = sdtw(y,y)
    for i in tqdm(range(x_n_samples)):
        xx_sdtw = sdtw(x[[i]],x[[i]])
        _x = x[[i]].repeat(y_n_samples, 1, 1)
        # distance = sdtw(_x, y)
        # distance = sdtw(_x,y) - 1/2*(sdtw(_x,_x)+sdtw(y,y))
        distance = sdtw(_x,y) - 1/2*(xx_sdtw+yy_sdtw)
        train_distance_matrix[i] = distance.cpu().numpy()
    return train_distance_matrix


def main():
    # Define dataset path
    data_dir = "./data/WHITEDv1.1"
    output_dir = "DTW_Classification_Results"
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    print('Loading data... ', end='')
    train_dataset = CustomDataset(data_dir=data_dir, split='train', resample=True, target_steps=512)
    test_dataset = CustomDataset(data_dir=data_dir, split='test', resample=True, target_steps=512)
    
    truncate = 10**5

    # Extract train and test data and labels
    train_data = train_dataset.x.squeeze(1).numpy()[:truncate]  # Remove singleton dimension
    train_labels = train_dataset.y.numpy()[:truncate]
    test_data = test_dataset.x.squeeze(1).numpy()[:truncate]   # Remove singleton dimension
    test_labels = test_dataset.y.numpy()[:truncate]

    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")
    
    # Compute DTW distance matrix for the training data
    print("Computing DTW distance matrix for training data...")
    
    x = torch.tensor(train_data).cuda()
    x = x[:,:,None]
    train_distance_matrix = calculate_sdtw_matrix(x, x)

    # train_distance_matrix = distance_matrix(train_data, parallel=True, show_progress=True) # frozen for some reasons...
    # train_distance_matrix = distance_matrix_fast(train_data) # frozen for some reasons...

    # Use k-NN classifier with DTW as the metric
    print("Training k-NN classifier with DTW...")
    knn = KNeighborsClassifier(n_neighbors=1, metric="precomputed")
    train_distance_matrix = np.clip(train_distance_matrix, a_min=0, a_max=np.inf)
    knn.fit(train_distance_matrix, train_labels)
    
    # Predict on the test data using DTW distances
    print("Computing DTW distance matrix between test and train data...")
    test_distance_matrix = np.zeros((len(test_data), len(train_data)))
    # for i, test_sample in tqdm(enumerate(test_data), total=len(test_data), desc="Processing"):
    #     for j, train_sample in enumerate(train_data):
    #         test_distance_matrix[i, j] = dtw.distance(test_sample, train_sample)
    x_test = torch.tensor(test_data).cuda()
    x_test = x_test[:,:,None]
    test_distance_matrix = calculate_sdtw_matrix(x_test, x)
    test_distance_matrix = np.clip(test_distance_matrix, a_min=0, a_max=np.inf)
    # pdb.set_trace()
    print("Predicting labels for test data...")
    y_pred_labels = knn.predict(test_distance_matrix)
    
    # Compute metrics
    print("Computing metrics...")
    accuracy = accuracy_score(test_labels, y_pred_labels)
    precision_macro = precision_score(test_labels, y_pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(test_labels, y_pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(test_labels, y_pred_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(test_labels, y_pred_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(test_labels, y_pred_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(test_labels, y_pred_labels, average='weighted', zero_division=0)

    # Micro-averaged metrics
    precision_micro = precision_score(test_labels, y_pred_labels, average='micro', zero_division=0)
    recall_micro = recall_score(test_labels, y_pred_labels, average='micro', zero_division=0)
    f1_micro = f1_score(test_labels, y_pred_labels, average='micro', zero_division=0)
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1-Score: {f1_micro:.4f}")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")

    # Save results to a file
    results_file = os.path.join(output_dir, "DTW_results.txt")
    with open(results_file, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {precision_macro:.4f}\n")
        file.write(f"Macro Recall: {recall_macro:.4f}\n")
        file.write(f"Macro F1-Score: {f1_macro:.4f}\n")
        file.write(f"Weighted Precision: {precision_weighted:.4f}\n")
        file.write(f"Weighted Recall: {recall_weighted:.4f}\n")
        file.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
        file.write(f"Micro Precision: {precision_micro:.4f}\n")
        file.write(f"Micro Recall: {recall_micro:.4f}\n")
        file.write(f"Micro F1-Score: {f1_micro:.4f}\n")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
