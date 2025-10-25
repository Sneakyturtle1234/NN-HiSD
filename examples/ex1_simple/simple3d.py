import os
import sys
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

sys.path.append(".../..")
import utils
from utils.networks import DNN

current_dir = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(41)

from functions import Potential_3d

def create_random_dataset(Region, num_points):
    np.random.seed(42)
    num_alpha = 10
    x = np.random.uniform(Region[0, 0], Region[0, 1], num_points * num_alpha)
    y = np.random.uniform(Region[1, 0], Region[1, 1], num_points * num_alpha)
    w = np.random.uniform(Region[2, 0], Region[2, 1], num_points * num_alpha)
    alpha = np.linspace(Region[3, 0], Region[3, 1], num_alpha)
    alpha = np.repeat(alpha, num_points)
    grid = np.column_stack((x, y, w, alpha))
    Z = torch.ones(num_points * num_alpha, dtype=torch.float64)
    for i in range(num_points * num_alpha):
        Z[i] = Potential_3d(grid[i, :])
    grid = torch.tensor(grid, dtype=torch.float64)

    return grid, Z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filename = 'parameters/parametric3d.pth'
model_path = os.path.join(current_dir, model_filename)
layer = [128, 128, 128]
if os.path.exists(model_path):
    try:
        model_info = torch.load(model_path)
        net = DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                    output_size=model_info.get('output_size')).double()
        net.load_state_dict(model_info['state_dict'])
        print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
        print(f"Model configuration - Input size: {model_info.get('input_size')}, "
                f"Layers: {model_info.get('layer_sizes')}, "
                f"Output size: {model_info.get('output_size')}")
    except Exception as e:
        net = DNN(input_size=4, layer_sizes=layer, output_size=1).double()
        net.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model (state_dict only).")
else:
    net = DNN(input_size=4, layer_sizes=layer, output_size=1).double()
    print("Construct model with structure: DNN")
    print(f"Model configuration - Input size: {np.shape(data)[-1]}, "
                f"Layers: {layer}, " f"Output size: {1}")

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.7)
num_points = 4000
New_Region = np.array([[-1, 7], [-1, 7], [-1, 7], [3, 7]])
batch_size = 1000

data, target = create_random_dataset(New_Region, num_points=num_points)
dataset = torch.utils.data.TensorDataset(data, target)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 30000
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    if (epoch + 1) % 10000 == 0:
        net.to('cpu')
        torch.save(net.state_dict(), model_path)
        net.to(device)

