import os
import torch
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
from sklearn.model_selection import train_test_split

def resample_time_series(data, target_steps=512):
    """
    Resample a time series to a fixed number of steps using linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        The original time-series data.

    target_steps : int
        The desired number of timesteps.

    Returns
    -------
    np.ndarray
        Resampled time series with the specified number of steps.
    """
    original_steps = len(data)
    original_indices = np.linspace(0, original_steps - 1, original_steps)
    target_indices = np.linspace(0, original_steps - 1, target_steps)
    resampled_data = np.interp(target_indices, original_indices, data)
    return resampled_data

def pad_or_truncate(sequence, target_length):
    """
    Pad or truncate a sequence to a fixed length.

    Parameters
    ----------
    sequence : np.ndarray
        The sequence to be padded or truncated.
    
    target_length : int
        The desired length of the sequence.

    Returns
    -------
    np.ndarray
        Padded or truncated sequence.
    """
    if len(sequence) > target_length:
        return sequence[:target_length]
    else:
        return np.pad(sequence, (0, target_length - len(sequence)), mode='constant')
def load_data(data_dir, resample=False, target_steps=None, test_size=0.2, val=False, val_size=0.1, random_state=42):
    """
    Load and preprocess time-series data, with optional validation split.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing FLAC files.
    resample : bool
        If True, resample all sequences to `target_steps`. Otherwise, pad or truncate.
    target_steps : int or None
        The target number of timesteps for resampling or padding/truncating.
    test_size : float
        Proportion of the dataset to include in the test split.
    val : bool
        If True, split the training data into a validation set.
    val_size : float
        Proportion of the training data to use as the validation set (only relevant if `val=True`).
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    If val=False:
        x_train, y_train, x_test, y_test, max_length, n_classes
    If val=True:
        x_train, y_train, x_val, y_val, x_test, y_test, max_length, n_classes
    """
    data_dict = {}

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".flac"):
            key = file_name.split("_")[0]
            file_path = os.path.join(data_dir, file_name)
            data, samplerate = sf.read(file_path)
            instantaneous_power = data[:, 0] * data[:, 1]

            if key not in data_dict:
                data_dict[key] = []

            data_dict[key].append(instantaneous_power)

    label_to_int = {label: idx for idx, label in enumerate(data_dict.keys())}
    x, y = [], []

    # Find maximum length if resample is False and target_steps is None
    max_length = 0
    if not resample and target_steps is None:
        for timeseries_list in data_dict.values():
            for timeseries in timeseries_list:
                max_length = max(max_length, len(timeseries))
        target_steps = max_length  # Use max length as target_steps for padding/truncating

    for label, timeseries_list in data_dict.items():
        for timeseries in timeseries_list:
            if resample:
                x.append(resample_time_series(timeseries, target_steps))
            else:
                x.append(pad_or_truncate(timeseries, target_steps))
            y.append(label_to_int[label])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Split data into train, validation, and test sets
    if val:
        # Combine validation and test sizes
        combined_split = test_size + val_size

        # First split: Separate combined test+validation set from the training set
        x_train, x_combined, y_train, y_combined = train_test_split(
            x, y, test_size=combined_split, random_state=random_state
        )

        # Second split: Separate validation and test sets
        val_ratio = val_size / combined_split
        x_val, x_test, y_val, y_test = train_test_split(
            x_combined, y_combined, test_size=1 - val_ratio, random_state=random_state
        )

        print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")
        print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

        return x_train, y_train, x_val, y_val, x_test, y_test, max_length, len(label_to_int)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
        print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        return x_train, y_train, x_test, y_test, max_length, len(label_to_int)

class CustomDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', resample=False, target_steps=512, test_size=0.2, val=False, val_size=0.1, random_state=42):
        if val:
            x_train, y_train, x_val, y_val, x_test, y_test, max_length, n_classes = load_data(
                data_dir, resample=resample, target_steps=target_steps, test_size=test_size, val=val, val_size=val_size, random_state=random_state
            )
        else:
            x_train, y_train, x_test, y_test, max_length, n_classes = load_data(
                data_dir, resample=resample, target_steps=target_steps, test_size=test_size, random_state=random_state
            )

        if split == 'train':
            self.x = torch.unsqueeze(torch.from_numpy(x_train), dim=1)
            self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        elif split == 'val' and val:
            self.x = torch.unsqueeze(torch.from_numpy(x_val), dim=1)
            self.y = torch.from_numpy(y_val).type(torch.LongTensor)
        elif split == 'test':
            self.x = torch.unsqueeze(torch.from_numpy(x_test), dim=1)
            self.y = torch.from_numpy(y_test).type(torch.LongTensor)
        else:
            raise AttributeError("Invalid split specified. Must be 'train', 'val', or 'test'.")

        self.max_length = max_length
        self.n_classes = n_classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        # return self.x[0], self.y[0]
        # return self.x[int(idx%100)], self.y[int(idx%100)]

