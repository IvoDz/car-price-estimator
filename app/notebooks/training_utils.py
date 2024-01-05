import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def plot_learning_curve(train_losses, test_losses, title='Learning Curve'):
    """
    Plot the learning curve.

    Parameters:
    - train_losses: List of training losses for each epoch
    - test_losses: List of test (validation) losses for each epoch
    - title: Title of the plot (default is 'Learning Curve')
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
    
def train_and_get_losses(learning_rate, batch_size, model, X_train, X_test, y_train, y_test):
    train_losses , test_losses = [], []
    learning_rate = 0.001
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    for epoch in range(100):
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
        print(f'Epoch [{epoch + 1}/100], Training Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}')

    return train_losses, test_losses