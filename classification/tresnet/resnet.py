import numpy as np
from os.path import join
import torch
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential, Softmax, AvgPool1d, Linear, CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm, trange
from typing import Tuple, Union, Callable, Optional
import random
import pdb 
import torch.nn as nn
import torchvision.models as models

def accuracy(y_true, y_pred):
    """Computes accuracy
    """
    return np.mean(y_true == y_pred)

class ResNet(Module):
    """
    Parameters
    ---------- 
    input_shape: tuple
        Shape of the input time series. (f, T) where 
        f is the number of features and T is the number
        of time steps.
    n_classes: int
        Number of classes. 
    n_feature_maps: int
        Number of feature maps to use. 
    name: str
        Model name. Default: "ResNet"
    verbose: int
        Controls verbosity while training and loading the models. 
    load_weights: bool
        If true load the weights of a previously saved model. 
    random_seed: int
        Random seed to instantiate the random number generator. 
    """
    def __init__(self, 
                 input_shape:Tuple[int, int], 
                 n_classes:int=2, 
                 n_feature_maps:int=64, 
                 name:str="resnet", 
                 verbose:int=0, 
                 load_weights:bool=False, 
                 random_seed:int=13,
                 multilabel=False):
        super().__init__()

        self.n_feature_maps = n_feature_maps
        self.name = name
        self.input_shape = input_shape # (num_features, num_timesteps)
        self.n_classes = n_classes
        self.verbose = verbose
        self.random_seed = random_seed
        self.set_all_seeds() # Control determinism
        self.multilabel = multilabel

        if load_weights: # Then just load the weights of the model.
            path = join(self.output_directory, f'best_model_{self.name}.hdf5')
            print(f"Loading model at {path}")
            self.model = torch.load(path)

            if self.verbose:
                print(self.model)

        else:
            # self.model = self.build_model()
            self.model = models.resnet18(pretrained=False, num_classes=self.n_classes)

    
    def set_all_seeds(self):
        random.seed(self.random_seed)
        # os.environ('PYTHONHASHSEED') = str(seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_model(self):
        # BLOCK 1
        self.conv_block_1 = Sequential(
            Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps)
            )
        
        self.shortcut_block_1 = Sequential(
            Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=1, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps)
            )

        # BLOCK 2
        self.conv_block_2 = Sequential(
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # expand channels for the sum
        self.shortcut_block_2 = Sequential(
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=1, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # BLOCK 3
        self.conv_block_3 = Sequential(
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # no need to expand channels because they are equal
        self.shortcut_block_3 = BatchNorm1d(num_features=self.n_feature_maps * 2)

        self.output_layer = Sequential(
            Linear(in_features=self.n_feature_maps * 2, out_features=self.n_classes),
            # Softmax(dim=1)
            )
    def forward(self, x):
        # pdb.set_trace()
        x = x[:,None,:,:]
        x = torch.concatenate([x,x,x],dim=1)
        x = self.model(x)

        if self.multilabel:
            x = F.sigmoid(x)
        return x 
    # def forward(self, x):
    #     output_block_1 = self.conv_block_1(x) + self.shortcut_block_1(x)
    #     output_block_1 = ReLU()(output_block_1)

    #     output_block_2 = self.conv_block_2(output_block_1) + self.shortcut_block_2(output_block_1)
    #     output_block_2 = ReLU()(output_block_2)

    #     output_block_3 = self.conv_block_3(output_block_2) + self.shortcut_block_3(output_block_2)
    #     output_block_3 = ReLU()(output_block_3)

    #     # output_gap_layer = AvgPool1d((self.input_shape[1], 1))(output_block_3).squeeze()
    #     output_gap_layer = AvgPool1d(kernel_size=output_block_3.shape[2], stride=1)(output_block_3).squeeze()

    #     preds = self.output_layer(output_gap_layer)

    #     # del output_block_1, output_block_2, output_block_3, gap_layer # release memory
        
    #     return preds

    def save_model(self, save_dir):
        with open(join(save_dir, f'{self.name}.pth'), 'wb') as f:
            torch.save(self.model, f)
    
   
def train(model:Callable, 
          train_data:Callable, 
          val_data:Optional[Callable],
          device:Callable=torch.device('cuda:0'), 
          batch_size:int=64,
          n_epochs:int=100,
          max_learning_rate:float=1e-3,
          early_stopping:bool=True,
          patience:int=2,
          tolerance:float=1e-3,
          save_dir:str='/home/scratch/mgoswami/',
          verbose:int=1,
          ):
    
    """
    Parameters
    ---------- 
    
    """
    if model.multilabel:
        loss_fn = nn.BCELoss(reduction='mean')  # Binary Cross-Entropy Loss for multi-label classification
    else:
        loss_fn = CrossEntropyLoss(reduction='mean') 
    optimizer = torch.optim.Adam(model.parameters(), lr=max_learning_rate)  
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.to(device) # send model to the device
    
    performance_metrics = {
        'val_acc': [],
        'val_loss': [],
        'train_acc': [],
        'train_loss': []
    }
    patience_counter = 0 # Counter for early stopping

    for epoch in trange(0, n_epochs, desc='Epochs'):
        loss_per_batch = []
        acc_per_batch = []
        for batch_data in train_dataloader:
            batch_x, batch_y = batch_data[0].to(device), batch_data[1].to(device)
            
            optimizer.zero_grad()
            
            batch_y_pred = model(batch_x)
            if model.multilabel:
                batch_y[batch_y==-1] = 0.
                batch_y = batch_y.to(batch_y_pred.dtype)
            loss = loss_fn(batch_y_pred, batch_y)

            loss.backward()
            optimizer.step()

            if model.multilabel:
                acc_per_batch.append(accuracy(batch_y.cpu().detach().numpy(), batch_y_pred.round().cpu().detach().numpy()))
            else:
                acc_per_batch.append(accuracy(batch_y.cpu().detach().numpy(), torch.argmax(batch_y_pred, dim=1).cpu().detach().numpy()))
            loss_per_batch.append(loss.cpu().detach().item())

        epoch_acc = np.mean(acc_per_batch)
        epoch_loss = np.mean(loss_per_batch)
        
        if verbose: print(f'Epoch {epoch} | Train Accuracy: {epoch_acc} | Loss: {epoch_loss}')
        
        performance_metrics['train_acc'].append(epoch_acc)
        performance_metrics['train_loss'].append(epoch_loss)

        if val_data is not None: 
            with torch.no_grad():
                y_true, y_pred = test(model=model, test_data=val_data, device=device, batch_size=batch_size)
                
                if model.multilabel:
                    y_true[y_true==-1] = 0.
                    y_true = y_true.astype(y_pred.dtype)
                    epoch_acc = accuracy(y_true, np.round(y_pred))
                else:
                    epoch_acc = accuracy(y_true, np.argmax(y_pred, axis=1))
                epoch_loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_true)).cpu().detach().item()
            
            performance_metrics['val_acc'].append(epoch_acc)
            performance_metrics['val_loss'].append(epoch_loss)

            if epoch > 1 and early_stopping:
                if performance_metrics['val_loss'][-2] + tolerance <= epoch_loss:
                    print('Loss less than tolerance!')
                    patience_counter = patience_counter + 1
                else:
                    print('Loss more than tolerance!')
                    patience_counter = 0
                
                if patience_counter >= patience:
                    print('Stopping early...')
                    break
            
            if verbose: print(f'Epoch {epoch} | Validation Accuracy: {epoch_acc} | Loss: {epoch_loss} | Patience counter: {patience_counter}')
                            
    # TODO: Save the best model 
    # model.save_model(save_dir)

    return model

def test(model:Callable, 
         test_data:Callable, 
         device:Callable=torch.device('cuda:0'), 
         batch_size:int=64):
    
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    y_preds = []
    y_true = []
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            batch_x, batch_y = batch_data[0].to(device), batch_data[1].to(device)                                     
            batch_y_pred = model(batch_x)

            y_preds.extend(batch_y_pred.cpu().detach().tolist())
            y_true.extend(batch_y.cpu().detach().tolist())
                    
    return np.asarray(y_true), np.asarray(y_preds)
