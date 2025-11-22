import os
import soundfile as sf  # To handle FLAC files
from whited_utils import resample_all_data, ClassificationDatasetWithMask # Import utility functions
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
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
import pdb 

import pandas as pd 
from torch.utils.data import Dataset
class Small_whited(Dataset):
    def __init__(self, data):
        self.data = data 
        self.seq_len = 512
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        timeseries_len = len(self.data[index][0])
        input_mask = np.ones(self.seq_len)
        input_mask[:self.seq_len - timeseries_len] = 0
        return torch.tensor(self.data[index][0])[None,:], torch.tensor(input_mask), torch.tensor(self.data[index][1])
    

# Path to the folder containing FLAC files
folder_path = "./data/WHITEDv1.1"
output_dir = "./output"
def process_flac_files(folder_path):
    """
    Process all FLAC files in a folder and extract instantaneous power.
    Returns a dictionary of time series data keyed by file identifier.
    """
    data_dict = {}  # Dictionary to store time-series data

    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".flac"):  # Process only FLAC files
                key = file_name.split("_")[0]  # Extract key from file name
                file_path = os.path.join(folder_path, file_name)  # Full file path

                # Load the FLAC file
                data, _ = sf.read(file_path)

                # Calculate instantaneous power (V * I)
                instantaneous_power = data[:, 0] * data[:, 1]

                # Store the instantaneous power in the dictionary
                if key not in data_dict:
                    data_dict[key] = []  # Initialize a list for this key
                data_dict[key].append(instantaneous_power)

        print(f"Processed {len(data_dict)} unique keys.")
        return data_dict

    except Exception as e:
        print(f"Error processing files: {e}")
        return {}


def get_embedding(model, dataloader, model_name):
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to("cuda").float()
            batch_masks = batch_masks.to("cuda")
            if model_name == 'moment':
                output = model(x_enc=batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]
                embedding = output.embeddings
                embeddings.append(embedding.detach().cpu().numpy())
            elif model_name == 'chronos':
                embedding = []
                for b in batch_x:
                    # _embedding = model(b[0][:128].cpu())[0]
                    _embedding = model(b[0].cpu())[0]
                    embedding.append(_embedding)
                embedding = torch.stack(embedding).to(torch.float32)
                # average the embedding over sequence length
                embedding = embedding.mean(dim=2)
                # # take the last embedding over sequence length
                # embedding = embedding[:,:,:,-1]
                embedding = embedding.reshape(embedding.shape[0],-1)
                embeddings.append(embedding.detach().cpu().numpy())
            elif model_name == 'ts2vec':
                batch_x = batch_x.cpu().transpose(1,2).numpy()
                embedding = model.encode(batch_x, encoding_window='full_series')    # (6368, 320)
                embeddings.append(embedding)
            labels.append(batch_labels)        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="chronos", help="The type of models we are testing (chronos | moment)"
    )
    parser.add_argument(
        "--save_embeddings", action="store_true", help="Save embeddings to .npy files"
    )
    args = parser.parse_args()
    # model_name = "moment"
    model_name = args.model
    
    # dataset = "small"
    dataset = "original"

    # use_nn = True
    use_nn = False
    print(model_name, "whited version: ", dataset, "use nn:", use_nn)

    if dataset == "small":
        data_tr = pd.read_pickle('data/whited_single_train_data.pkl')
        data_test = pd.read_pickle('data/whited_single_train_data.pkl')
        train_dataset = Small_whited(data_tr)
        test_dataset = Small_whited(data_test)
    else:
        # Resample all time series to a target number of steps
        target_steps = 512
        # Process FLAC files to extract time series data
        data_dict = process_flac_files(folder_path)
        resampled_data_dict = resample_all_data(data_dict, target_steps)

        # Print resampled data details
        for key, timeseries_list in resampled_data_dict.items():
            print(f"Key: {key}")
            for idx, timeseries in enumerate(timeseries_list):
                print(f"  Resampled Timeseries {idx + 1}: {timeseries.shape} steps")

        # Create datasets
        train_dataset = ClassificationDatasetWithMask(resampled_data_dict, data_split='train')
        test_dataset = ClassificationDatasetWithMask(resampled_data_dict, data_split='test')

    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # Verify data format
    for batch_x, batch_masks, batch_labels in train_dataloader:
        print(f"Batch X Shape: {batch_x.shape}")  # Should be (batch_size, 1, seq_len)
        print(f"Batch Masks Shape: {batch_masks.shape}")  # Should be (batch_size, seq_len)
        print(f"Batch Labels Shape: {batch_labels.shape}")  # Should be (batch_size,)
        break
    
    if model_name == "moment":
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode
        )
        model.init()

        model.to("cuda").float()
    elif model_name == "chronos":
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        model = pipeline.embed
    elif model_name == 'ts2vec':
        import sys
        sys.path.append('/home/prquan/Github/TS-foundation-model/classification/ts2vec/')
        from ts2vec import TS2Vec
        ts2vec_model = TS2Vec(input_dims=1, device="cuda", batch_size=8, lr=0.001, output_dims=320)
        # Extract and reshape train and test time series data
        train_timeseries = np.stack([item[0].squeeze().numpy() for item in train_dataset])  # Remove singleton dims
        train_timeseries = train_timeseries[:, :, np.newaxis]  # Ensure shape: (num_samples, seq_len, num_features)
        test_timeseries = np.stack([item[0].squeeze().numpy() for item in test_dataset])  # Remove singleton dims
        test_timeseries = test_timeseries[:, :, np.newaxis]  # Ensure shape: (num_samples, seq_len, num_features)
        ts2vec_model.fit(train_timeseries, n_epochs=100, n_iters=100) # (25471, 512, 1)
        model = ts2vec_model

    # if model_name == 'ts2vec':
    if False:
        train_embeddings = np.load('tr.npy')
        test_embeddings = np.load('test.npy')
        train_labels = np.load('tr_label.npy')
        test_labels = np.load('test_label.npy')
    else:
        train_embeddings, train_labels = get_embedding(model, train_dataloader, model_name)
        test_embeddings, test_labels = get_embedding(model, test_dataloader, model_name)

    print(train_embeddings.shape, train_labels.shape)
    print(test_embeddings.shape, test_labels.shape)

    # Save embeddings if requested
    if args.save_embeddings:
        np.save(f"{model_name}_train_embeddings_whited.npy", train_embeddings)
        np.save(f"{model_name}_train_labels_whited.npy", train_labels)
        np.save(f"{model_name}_test_embeddings_whited.npy", test_embeddings)
        np.save(f"{model_name}_test_labels_whited.npy", test_labels)
        print(f"Embeddings saved:")
        print(f"  - {model_name}_train_embeddings_whited.npy: {train_embeddings.shape}")
        print(f"  - {model_name}_train_labels_whited.npy: {train_labels.shape}")
        print(f"  - {model_name}_test_embeddings_whited.npy: {test_embeddings.shape}")
        print(f"  - {model_name}_test_labels_whited.npy: {test_labels.shape}")

    if use_nn:
        from fcn import train_fcn_classifier_single_class
        y_pred_train, y_pred_test, train_accuracy, test_accuracy = train_fcn_classifier_single_class(train_embeddings, train_labels, test_embeddings, test_labels)
    else:
        if model_name == 'moment':
        # if False:
            clf = fit_svm(features=train_embeddings, y=train_labels)
        elif model_name in ('chronos', 'ts2vec', 'moment'):
            # clf = SVC()  # You can specify kernel='linear', 'rbf', etc., as needed
            # clf.fit(train_embeddings, train_labels)

            from sklearn.model_selection import GridSearchCV, train_test_split

            def fit_svm(features, y, MAX_SAMPLES=10000):
                nb_classes = np.unique(y, return_counts=True)[1].shape[0]
                train_size = features.shape[0]

                svm = SVC(C=np.inf, gamma='scale')
                if train_size // nb_classes < 5 or train_size < 50:
                    return svm.fit(features, y)
                else:
                    grid_search = GridSearchCV(
                        svm, {
                            'C': [
                                0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                                1e6
                            ],
                            'kernel': ['rbf'],
                            'degree': [3],
                            'gamma': ['scale'],
                            'coef0': [0],
                            'shrinking': [True],
                            'probability': [False],
                            'tol': [0.001],
                            'cache_size': [200],
                            'class_weight': [None],
                            'verbose': [False],
                            'max_iter': [10000000],
                            'decision_function_shape': ['ovr'],
                            'random_state': [None]
                        },
                        cv=5, n_jobs=5
                    )
                    # If the training set is too large, subsample MAX_SAMPLES examples
                    if train_size > MAX_SAMPLES:
                        split = train_test_split(
                            features, y,
                            train_size=MAX_SAMPLES, random_state=0, stratify=y
                        )
                        features = split[0]
                        y = split[2]
                        
                    grid_search.fit(features, y)
                    return grid_search.best_estimator_

            clf = fit_svm(train_embeddings, train_labels)

        y_pred_train = clf.predict(train_embeddings)
        y_pred_test = clf.predict(test_embeddings)
        train_accuracy = clf.score(train_embeddings, train_labels)
        test_accuracy = clf.score(test_embeddings, test_labels)
    # pdb.set_trace()
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("y_pred_test", y_pred_test)
    #take argmax of y_pred_test
    # y_pred_labels=np.argmax(y_pred_test, axis=1)
    y_pred_labels=y_pred_test
    print("Computing metrics...")
    accuracy = accuracy_score(test_labels, y_pred_labels)
    precision_macro = precision_score(test_labels, y_pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(test_labels, y_pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(test_labels, y_pred_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(test_labels, y_pred_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(test_labels, y_pred_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(test_labels, y_pred_labels, average='weighted', zero_division=0)

    print(model_name, "whited version: ", dataset, "use nn:", use_nn)
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")

    # Save results to a file
    results_file = os.path.join(output_dir, "Moment_results.txt")
    with open(results_file, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {precision_macro:.4f}\n")
        file.write(f"Macro Recall: {recall_macro:.4f}\n")
        file.write(f"Macro F1-Score: {f1_macro:.4f}\n")
        file.write(f"Weighted Precision: {precision_weighted:.4f}\n")
        file.write(f"Weighted Recall: {recall_weighted:.4f}\n")
        file.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
    print(f"Results saved to {results_file}")