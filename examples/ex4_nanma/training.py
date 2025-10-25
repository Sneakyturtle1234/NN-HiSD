import os
import torch
# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))


class CustomNet(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x


def create_dataset(datapath):
    data = np.loadtxt(datapath)
    angle = torch.tensor(data[:, 0:2], dtype=torch.float64)
    free_energy = torch.tensor(data[:, 2], dtype=torch.float64)

    return angle, free_energy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
net = CustomNet(input_size=2, layer_sizes=[128, 128, 128, 128, 128], output_size=1).to(device)
net.double()

model_filename = 'nanma.pth'
model_path = os.path.join(current_dir, model_filename)
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model.")

epochs = 20
Region = np.array([[-180, 180], [-180, 180]])
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)

datapath = os.path.join(current_dir, 'vacuum.pmf')

data, target = create_dataset(datapath)
data = data / 180
data, target = data.to(device), target.to(device)
dataset = torch.utils.data.TensorDataset(data, target)
batch_size = 1000
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

net.to('cpu')
torch.save(net.state_dict(), model_path)
print(f"Model parameters saved to {model_path}.")


with torch.no_grad():
    net.eval()
    grid, true_values = create_dataset(os.path.join(current_dir, 'vacuum.pmf'))
    X = grid[:, 0]
    Y = grid[:, 1]
    n = int(np.sqrt(len(X)))
    X = X.reshape((n, n))
    Y = Y.reshape((n, n))
    predicted_values = net(grid / 180).view(-1)
    error = true_values - predicted_values
    X, Y = X.numpy(), Y.numpy()
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('True Potential')
    plt.contour(X, Y, true_values.numpy().reshape(n, n), cmap='viridis', levels=30)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Learned Potential')
    plt.contour(X, Y, predicted_values.numpy().reshape(n, n), cmap='viridis', levels=30)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Error Heatmap')
    plt.contourf(X, Y, error.numpy().reshape(n, n), cmap='viridis',
                 levels=np.arange(-1, 1, 0.2) * 0.05)
    plt.colorbar()

    plt.show()

