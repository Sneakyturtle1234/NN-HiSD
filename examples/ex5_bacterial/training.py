import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import sys

sys.path.append(".../..")
import utils, core


def CreateData(Region):
    dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(dir, 'tsne_s28.npy'))
    value = np.loadtxt(os.path.join(dir, 'num.txt'))
    value = 13.7 - np.log(value)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    
    data = torch.tensor(data, dtype=torch.float64)
    value = torch.tensor(value, dtype=torch.float64)
    
    return data, value, mean, std

def Train_Model(layer, Region, batch_size=5, epochs=10000):
    # Create the DNN parameters
    model = utils.networks.DNN(in_dim, layer, out_dim)
    if os.path.exists('bacterial.pth'):
        model.load_state_dict(torch.load('bacterial.pth'))
    model.double()
    
    # Define the loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-3)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.7)

    # Create the dataloader and labels
    data, labels, mean, std = CreateData(Region)
    model.mean, model.std = mean, std
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    
    # Train the model
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')

    # Save the model parameters
    model.to('cpu')
    torch.save(model.state_dict(), 'bacterial.pth')


if __name__ == '__main__':
    # Setting the parameters
    [in_dim, out_dim] = [2, 1]
    layer = [128, 128, 128]
    Region = [[-250, 250], [-250, 250]]
    epochs = 5000
    
    Train_Model(layer, Region, batch_size=5, epochs=epochs)