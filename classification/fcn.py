import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pdb 
from tqdm import tqdm 
import numpy as np 

# Define the Fully-Connected Network (FCN)
class FCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # for multi-label classification
        )

    def forward(self, x):
        return self.model(x)

# Wrapper function to train and evaluate the FCN

def train_fcn_classifier(train_embeddings, train_labels, test_embeddings, multi_calss=True, epochs=20, batch_size=32, learning_rate=1e-3):
    # Convert embeddings and labels to tensors
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)

    # Create dataset and DataLoader
    dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model, loss, and optimizer
    input_dim = train_embeddings.shape[1]
    output_dim = train_labels.shape[1]
    model = FCNClassifier(input_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for batch_embeddings, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            if multi_calss:
                batch_labels[batch_labels==-1]=0
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch {} | loss {:3f}'.format(epoch, epoch_loss/len(dataloader)))
        

    # Evaluate predictions (probabilities)
    model.eval()
    with torch.no_grad():
        y_pred_train_proba = model(torch.tensor(train_embeddings, dtype=torch.float32)).numpy()
        y_pred_test_proba = model(torch.tensor(test_embeddings, dtype=torch.float32)).numpy()

    return y_pred_train_proba, y_pred_test_proba

from sklearn.metrics import accuracy_score

# Define the Fully-Connected Network (FCN)
class FCNClassifierSingleClass(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNClassifierSingleClass, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            # nn.Softmax(dim=1)  # for single-label classification
        )

    def forward(self, x):
        return self.model(x)

# Wrapper function to train and evaluate the FCN for single-label classification (labels range from 1 to N)
def train_fcn_classifier_single_class(train_embeddings, train_labels, test_embeddings, test_labels, epochs=200, batch_size=32, learning_rate=1e-3):
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)

    dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = train_embeddings.shape[1]
    output_dim = len(torch.unique(train_labels_tensor))
    model = FCNClassifierSingleClass(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(epochs)):
        avg_loss = []
        for batch_embeddings, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print('Epoch {} | loss {:3f}'.format(epoch, np.mean(avg_loss)))

    model.eval()
    with torch.no_grad():
        y_pred_train = model(train_embeddings_tensor).argmax(dim=1).numpy()
        y_pred_test = model(test_embeddings_tensor).argmax(dim=1).numpy()

    train_accuracy = accuracy_score(train_labels, y_pred_train)
    test_accuracy = accuracy_score(test_labels, y_pred_test)

    return y_pred_train, y_pred_test, train_accuracy, test_accuracy
