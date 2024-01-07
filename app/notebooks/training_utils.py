import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def plot_learning_curve(train_losses:list, test_losses:list, title:str='Learning Curve') -> None:
    """ 
    Plots learning curves for both given train and test losses.

    Args:
        train_losses (list): All train losses recorded during each epoch
        test_losses (list): All test losses recorded during each epoch
        title (str, optional): Title of the plot. Defaults to 'Learning Curve'.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def train_and_get_losses(learning_rate:int,
                         batch_size:int,
                         epochs:int,
                         model:nn.Module,
                         X_train:torch.tensor,
                         X_test:torch.tensor,
                         y_train:torch.tensor,
                         y_test:torch.tensor):
    """
    Given all necessary data and hyperparameters, train the model and save the losses during each epoch.

    Args:
        learning_rate (int): Learning rate for training
        batch_size (int): Batch size for training
        epochs (int): Nummber of epochs for training
        model (nn.Module): PyTorch model class -- neural network architecture
        X_train (torch.tensor): Training set
        X_test (torch.tensor): Testing set
        y_train (torch.tensor): Target column (training)
        y_test (torch.tensor):  Target column (testing)

    Returns:
        tuple: tuple containing two lists of losses, both test and train
    """
    print(f"Starting training : lr={learning_rate}, batch={batch_size}")
    train_losses , test_losses = [], []
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001) 

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()  
            outputs = model(batch_X)  
            loss = criterion(outputs, batch_y.view(-1, 1)) 

            loss.backward()  
            optimizer.step() 

            epoch_loss += loss.item()

        with torch.no_grad():
            model.eval()
            test_outputs = model(X_test)
            loss = criterion(test_outputs, y_test.view(-1, 1))
            model.train()

        train_losses.append(epoch_loss / len(X_train))
        test_losses.append(loss.item())
        
    return train_losses, test_losses