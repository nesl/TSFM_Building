import numpy as np
from tresnet.dataset import CustomDataset
from tresnet.resnet import ResNet, train, test, accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pdb 

# Assuming CustomDataset, ResNet, train, and test are properly defined elsewhere.
def main():
    data_dir = "./data/WHITEDv1.1"
    # Load datasets
    print('Loading data... ', end='')
    train_data = CustomDataset(data_dir=data_dir, split='train', target_steps=512, test_size=0.2, val=True, val_size=0.2)
    val_data = CustomDataset(data_dir=data_dir, split='val', target_steps=512, test_size=0.2, val=True, val_size=0.2)
    test_data = CustomDataset(data_dir=data_dir, split='test', target_steps=512, test_size=0.2, val=True, val_size=0.2)
    # test_data = CustomDataset(data_dir=data_dir, split='train', target_steps=512, test_size=0.2, val=True, val_size=0.2)
    print("shape of train data x and y", train_data.x.shape, train_data.y.shape)
    print("shape of test data x and y", test_data.x.shape, test_data.y.shape)
    #from the test data generate a valiation set
    # pdb.set_trace()

    n_classes = train_data.n_classes

    # Initialize the ResNet model with input shape and output classes
    clf = ResNet(input_shape=(1, 512), n_feature_maps=1, n_classes=n_classes)
    # clf.build_model()
    # Train the model
    train(
        model=clf,
        train_data=train_data,
        val_data=val_data, #val_dataset,
        batch_size=64,
        n_epochs=1000,
        max_learning_rate=1e-3,
        device='cuda',
        save_dir='',  # Specify save directory if needed
        early_stopping = True,
    )

    # Test the model
    y_true, y_preds = test(
        model=clf,
        test_data=test_data,
        batch_size=64,
        device='cuda'
    )
    
    print("y_preds:", y_preds)  # Predicted probabilities
    print("y_true:", y_true)    # True labels

    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_preds, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision_macro = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")

    # Micro-averaged metrics
    precision_micro = precision_score(y_true, y_pred_labels, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred_labels, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred_labels, average='micro', zero_division=0)
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1-Score: {f1_micro:.4f}")

    # Save results to a file
    results_file = "ResNet_results.txt"
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

if __name__ == '__main__':
    main()
