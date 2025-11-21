import pandas as pd
import pickle
import zipfile
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except:
    print("Attempted to use torch but failed. Continue.")
from sklearn.model_selection import train_test_split
import pdb 
from tqdm import tqdm 

def resample_time_series(data, target_length=512):
    """
    Resample a time series to a fixed number of steps using linear interpolation.
    """
    original_length = len(data)
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    return np.interp(target_indices, original_indices, data)


def load_train_data(train_y_path, zip_file_path, seq_len=512, test_size=0.2, random_state=42):
    """
    Load and resample training data, and split into train and test sets.
    """
    # Load training labels
    df_train_y = pd.read_csv(train_y_path, index_col=0)
    print(f"Loaded training labels. Number of samples: {len(df_train_y)}")
    resampled_data_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        pkl_files = zip_ref.namelist()
        print(f"Number of .pkl files in ZIP: {len(pkl_files)}")

        for idx, row in df_train_y.iterrows():
            filename = row.name
            if not filename.endswith('.pkl'):
                filename = f"{filename}.pkl"

            pkl_file = f"train_X/{filename}"
            labels = row.values.astype(np.int64)  # Ensure labels are integers

            if pkl_file in pkl_files:
                with zip_ref.open(pkl_file, 'r') as f:
                    data = pickle.load(f)
                    resampled_values = resample_time_series(data['v'], target_length=seq_len)
                    resampled_data_list.append({
                        "timeseries": resampled_values,
                        "labels": labels
                    })
            else:
                print(f"File not found in ZIP: {pkl_file}")

    print(f"Number of resampled samples: {len(resampled_data_list)}")
    if len(resampled_data_list) == 0:
        raise ValueError("No matching files found between the labels and ZIP contents.")

    # Split into train and test sets
    train_data, test_data = train_test_split(
        resampled_data_list, test_size=test_size, random_state=random_state
    )
    return train_data, test_data

def load_test_data(zip_file_path, prefix, seq_len=512):
    test_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        pkl_files = zip_ref.namelist()
        pkl_files = [p for p in pkl_files if p.endswith('.pkl')]
        print(f"Number of .pkl files in ZIP: {len(pkl_files)}")
        for filename in tqdm(pkl_files):
            if not filename.endswith('.pkl'):
                continue
            with zip_ref.open(filename, 'r') as f:
                data = pickle.load(f)
                resampled_values = resample_time_series(data['v'], target_length=seq_len)
            test_list.append(resampled_values)
        test_list = np.array(test_list)
    return test_list, pkl_files

def compute_metrics(y_true, y_pred):
    """
    Computes precision, recall, F1 score, and accuracy for a multi-label classification task.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        Ground truth labels (-1, 0, 1).
    y_pred : ndarray of shape (n_samples, n_labels)
        Predicted labels (-1, 1).

    Returns
    -------
    dict
        Dictionary containing precision, recall, F1 score, and accuracy.
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each sample and label
    for i in range(y_true.shape[0]):  # For each sample
        for j in range(y_true.shape[1]):  # For each label
            if y_true[i, j] == 0:  # Skip masked labels
                continue
            
            total_predictions += 1  # Count valid predictions
            if y_true[i, j] == y_pred[i, j]:  # Correct prediction
                correct_predictions += 1
                if y_true[i, j] == 1:  # True positive
                    true_positives += 1
            else:
                if y_true[i, j] == 1 and y_pred[i, j] == -1:  # False negative
                    false_negatives += 1
                elif y_true[i, j] == -1 and y_pred[i, j] == 1:  # False positive
                    false_positives += 1

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Return metrics
    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
    }


class CustomDataset(Dataset):
    def __init__(self, data, seq_len=512, zero_out=True):
        """
        Initialize the dataset.

        Parameters
        ----------
        data : list of dict
            A list where each entry contains 'timeseries' and 'labels'.
        seq_len : int
            Length of the time series sequences.
        """
        self.data = data
        self.seq_len = seq_len
        self.zero_out = zero_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data instance.
        """
        item = self.data[idx]
        timeseries = torch.tensor(item["timeseries"], dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(item["labels"], dtype=torch.long)
        
        if self.zero_out:
            labels[labels==0] = 1.0 # make zero labels as the true labels

        return timeseries, labels
