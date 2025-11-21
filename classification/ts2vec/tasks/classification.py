import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb 

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    print(f"Train labels distribution: {np.unique(train_labels, return_counts=True)}")
    print(f"Test labels distribution: {np.unique(test_labels, return_counts=True)}")

    # Encode data
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # Debugging embeddings
    print(f"Train embeddings: mean={np.mean(train_repr)}, std={np.std(train_repr)}")
    print(f"Test embeddings: mean={np.mean(test_repr)}, std={np.std(test_repr)}")

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    np.save('tr.npy', train_repr)
    np.save('test.npy', test_repr)
    np.save('tr_label.npy', train_labels)
    np.save('test_label.npy', test_labels)

    clf = fit_clf(train_repr, train_labels)

    # Predictions
    y_pred = clf.predict(test_repr)

    # Compute metrics
    metrics = {
        'precision_macro': precision_score(test_labels, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(test_labels, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(test_labels, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(test_labels, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(test_labels, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(test_labels, y_pred, average='weighted', zero_division=0),
    }

    return y_pred, metrics
