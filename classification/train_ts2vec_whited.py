
import os
import sys
import time
import datetime
import numpy as np
import torch
from tresnet.dataset import CustomDataset
from tresnet.resnet import ResNet, train, test, accuracy
import pdb 

# Add the ts2vec folder to sys.path
sys.path.append('./ts2vec')

# Import utilities and other required modules from ts2vec
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import tasks
import datautils
from ts2vec import TS2Vec
import matplotlib.pyplot as plt

def main():
    # Define dataset path
    data_dir = "./data/WHITEDv1.1"
    # Load datasets
    print('Loading data... ', end='')
    train_dataset = CustomDataset(data_dir=data_dir, split='train', resample=True, target_steps=512)
    test_dataset = CustomDataset(data_dir=data_dir, split='test', resample=True, target_steps=512)
    n_classes = train_dataset.n_classes

    # Extract train and test data and labels
    train_data = train_dataset.x.squeeze(1).numpy()  # Remove singleton dimension
    train_labels = train_dataset.y.numpy()
    test_data = test_dataset.x.squeeze(1).numpy()   # Remove singleton dimension
    test_labels = test_dataset.y.numpy()

    print(f"Train data shape after squeezing: {train_data.shape}")
    print(f"Test data shape after squeezing: {test_data.shape}")

    # Debugging: Ensure all samples have the expected length
    for i, sample in enumerate(train_data):
        if sample.shape[0] != 512:
            print(f"Sample {i} has unexpected length: {sample.shape[0]}")

    # Check for samples with insufficient length
    min_length = 512
    invalid_train_samples = [i for i, sample in enumerate(train_data) if sample.shape[0] < min_length]
    invalid_test_samples = [i for i, sample in enumerate(test_data) if sample.shape[0] < min_length]

    if invalid_train_samples or invalid_test_samples:
        raise ValueError(
            f"Invalid samples detected: Train samples: {invalid_train_samples}, Test samples: {invalid_test_samples}. "
            f"Each sample must have exactly {min_length} timesteps. Check preprocessing."
        )

    print('All samples are valid.')

    # Initialize device
    device = init_dl_program("cpu", seed=42, max_threads=None)

    # Task type
    task_type = 'classification'

    # Configurations
    config = dict(
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000
    )
    save_every = None

    # Directory for saving outputs
    run_dir = 'training/WHITED'
    os.makedirs(run_dir, exist_ok=True)

    # Initialize and train TS2Vec model
    print('Initializing TS2Vec model...')
    t = time.time()
    model = TS2Vec(
        input_dims=1,
        device=device,
        **config
    )
    train_data = train_data[:, :, np.newaxis]  # Expanding last dimension to add feature
    test_data = test_data[:, :, np.newaxis]    # Same for test data
    print(f"New train data shape: {train_data.shape}")  # Expected: (1071, 512, 1)
    print(f"New test data shape: {test_data.shape}")    # Expected: (268, 512, 1)
    
    
    print('Training TS2Vec model...')
    loss_log = model.fit(
        train_data,
        n_epochs=100,
        n_iters=100,
        verbose=True
    )
    print('Save model weights')
    torch.save(model.net.state_dict(), 'model_weights.pth')

    # Save the trained model
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    # Evaluate the model
    eval = True
    if eval:
        print('Evaluating model...')
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(
                model, train_data, train_labels, test_data, test_labels, eval_protocol='svm'
            )
        
        # Save evaluation results
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)
        print('Accu: {:.4f}'.format((out==test_labels).sum()/len(test_labels)))
        for k in eval_res:
            print('{}: {:.4f}'.format(k, eval_res[k]))

if __name__ == '__main__':
    main()



   