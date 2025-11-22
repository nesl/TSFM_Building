import os
import numpy as np
import pandas as pd
import pickle
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import foundation models
try:
    from momentfm.models.statistical_classifiers import fit_svm
    from momentfm import MOMENTPipeline
except:
    pass
try:
    from chronos import ChronosPipeline
except:
    pass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC


def resample_time_series(time_series, target_length=512):
    """
    Resamples a time series to the target length using linear interpolation.

    Parameters:
    - time_series: Original time series values
    - target_length: Desired length for the resampled time series

    Returns:
    - Resampled time series of length target_length
    """
    original_length = len(time_series)
    if original_length == target_length:
        return time_series
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)
    interpolator = interp1d(original_indices, time_series, kind='linear', fill_value="extrapolate")
    return interpolator(target_indices)


def load_train_data(train_y_path, zip_file_path, seq_len=512, test_size=0.2, random_state=42):
    """
    Loads and resamples training data, then splits it into train and test sets.

    Parameters:
    - train_y_path: Path to the training target CSV file
    - zip_file_path: Path to the ZIP file containing the training .pkl files
    - seq_len: Desired sequence length for resampling
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed for reproducibility of the split

    Returns:
    - train_data, test_data: Train and test datasets with resampled timeseries and associated labels
    """
    df_train_y = pd.read_csv(train_y_path, index_col=0)
    print(f"Loaded training labels. Number of samples: {len(df_train_y)}")

    resampled_data_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        pkl_files = zip_ref.namelist()
        print(f"Number of .pkl files in ZIP: {len(pkl_files)}")

        for idx, row in df_train_y.iterrows():
            filename = row.name
            if filename.endswith('.pkl'):
                pkl_file = f"train_X/{filename}"
            else:
                pkl_file = f"train_X/{filename}.pkl"

            labels = row.values

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


class ClassificationDatasetWithMask(Dataset):
    """
    Dataset for multi-label classification with MOMENT model.
    """
    def __init__(self, data_list, seq_len=512):
        """
        Parameters:
        - data_list: List of dictionaries containing "timeseries" and "labels"
        - seq_len: Fixed sequence length for each time series
        """
        self.seq_len = seq_len
        self.data = []
        self.input_masks = []
        self.labels = []

        for entry in data_list:
            timeseries = entry["timeseries"]
            labels = entry["labels"]

            # Add channel dimension (1 for single-channel data)
            timeseries = np.expand_dims(timeseries, axis=0)

            # Create input mask
            input_mask = np.ones(seq_len)

            # Append to dataset
            self.data.append(timeseries)
            self.input_masks.append(input_mask)
            self.labels.append(labels)

        # Convert lists to NumPy arrays
        self.data = np.array(self.data)
        self.input_masks = np.array(self.input_masks)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
        - Tuple of (timeseries, input_mask, label)
        """
        timeseries = self.data[idx]
        input_mask = self.input_masks[idx]
        label = self.labels[idx]
        return (
            torch.tensor(timeseries, dtype=torch.float32),  # Shape: (1, seq_len)
            torch.tensor(input_mask, dtype=torch.float32),  # Shape: (seq_len,)
            torch.tensor(label, dtype=torch.float32),       # Shape: (num_labels,)
        )


def get_embedding(model, dataloader, model_name):
    """
    Generates embeddings using MOMENT or Chronos model.

    Parameters:
    - model: MOMENT or Chronos model
    - dataloader: DataLoader for the dataset
    - model_name: 'moment' or 'chronos'

    Returns:
    - embeddings: Array of shape (num_samples, embedding_dim)
    - labels: Array of shape (num_samples, num_labels)
    """
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to("cuda").float()
            batch_masks = batch_masks.to("cuda")

            if model_name == 'moment':
                output = model(x_enc=batch_x, input_mask=batch_masks)
                embedding = output.embeddings
                embeddings.append(embedding.detach().cpu().numpy())
            elif model_name == 'chronos':
                embedding = []
                for b in batch_x:
                    _embedding = model(b[0].cpu())[0]
                    embedding.append(_embedding)
                embedding = torch.stack(embedding).to(torch.float32)
                # Average the embedding over sequence length
                embedding = embedding.mean(dim=2)
                # Reshape to flatten
                embedding = embedding.reshape(embedding.shape[0], -1)
                embeddings.append(embedding.detach().cpu().numpy())

            labels.append(batch_labels)

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="moment", help="The type of models we are testing (chronos | moment)"
    )
    parser.add_argument(
        "--save_embeddings", action="store_true", help="Save embeddings to .npy files"
    )
    args = parser.parse_args()

    model_name = args.model
    print(f"Running {model_name.upper()} on BTS dataset")

    # File paths for BTS dataset
    zip_file_path = 'data/train_X_v0.1.0.zip'
    train_y_path = 'data/train_y_v0.1.0.csv'
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Load BTS data (same as notebook)
    print('Loading BTS data...')
    train_data, test_data = load_train_data(train_y_path, zip_file_path, seq_len=512)

    # Create datasets with masks (same as notebook)
    train_dataset = ClassificationDatasetWithMask(train_data)
    test_dataset = ClassificationDatasetWithMask(test_data)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # Verify data format
    for batch_x, batch_masks, batch_labels in train_dataloader:
        print(f"Batch X Shape: {batch_x.shape}")  # Should be (batch_size, 1, seq_len)
        print(f"Batch Masks Shape: {batch_masks.shape}")  # Should be (batch_size, seq_len)
        print(f"Batch Labels Shape: {batch_labels.shape}")  # Should be (batch_size, num_labels)
        break

    # Load model
    if model_name == "moment":
        print("Loading MOMENT-1-large model...")
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={'task_name': 'embedding'},
        )
        model.init()
        model.to("cuda").float()
    elif model_name == "chronos":
        print("Loading Chronos-T5-Large model...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        model = pipeline.embed

    # Generate embeddings
    print("\nGenerating embeddings...")
    train_embeddings, train_labels = get_embedding(model, train_dataloader, model_name)
    test_embeddings, test_labels = get_embedding(model, test_dataloader, model_name)

    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Save embeddings if requested
    if args.save_embeddings:
        np.save(f"{model_name}_train_embeddings_bts.npy", train_embeddings)
        np.save(f"{model_name}_train_labels_bts.npy", train_labels)
        np.save(f"{model_name}_test_embeddings_bts.npy", test_embeddings)
        np.save(f"{model_name}_test_labels_bts.npy", test_labels)
        print(f"\nEmbeddings saved:")
        print(f"  - {model_name}_train_embeddings_bts.npy: {train_embeddings.shape}")
        print(f"  - {model_name}_train_labels_bts.npy: {train_labels.shape}")
        print(f"  - {model_name}_test_embeddings_bts.npy: {test_embeddings.shape}")
        print(f"  - {model_name}_test_labels_bts.npy: {test_labels.shape}")

    # Train SVM classifier (for multi-label, we train one classifier per label)
    print("\n" + "="*60)
    print("Training SVM classifiers for multi-label classification...")
    print("="*60)
    n_labels = train_labels.shape[1]

    # For multi-label classification, train one SVM per label
    predictions = []
    accuracies = []

    for label_idx in tqdm(range(n_labels), desc="Training classifiers"):
        # Get labels for this specific class (handle -1, 0, 1 encoding)
        train_y = train_labels[:, label_idx]
        test_y = test_labels[:, label_idx]

        # Filter out samples with 0 (masked/unknown labels)
        train_mask = train_y != 0
        test_mask = test_y != 0

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            # Skip labels with too few samples
            continue

        # Train SVM for this label
        clf = SVC(C=1.0, kernel='rbf', gamma='scale')
        clf.fit(train_embeddings[train_mask], train_y[train_mask])

        # Predict on test set
        y_pred = clf.predict(test_embeddings[test_mask])
        accuracy = accuracy_score(test_y[test_mask], y_pred)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies) if accuracies else 0.0

    print(f"\n" + "="*60)
    print(f"{model_name.upper()} Results on BTS dataset (multi-label)")
    print("="*60)
    print(f"Average per-label accuracy: {avg_accuracy:.4f}")
    print(f"Number of labels evaluated: {len(accuracies)} / {n_labels}")

    # Save results to a file
    results_file = os.path.join(output_dir, f"{model_name}_bts_results.txt")
    with open(results_file, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Dataset: BTS (multi-label classification)\n")
        file.write(f"Average per-label accuracy: {avg_accuracy:.4f}\n")
        file.write(f"Number of labels evaluated: {len(accuracies)} / {n_labels}\n")
    print(f"\nResults saved to {results_file}")
