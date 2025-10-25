import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.networks import DNN, custom_loss
from utils import calculate_derivatives

def trainer(data, target, model_path, layer, learning_rate=1e-3, seed=42,
            epochs=30000, batch_size=10, report=True, grad_output=False, 
            report_interval=10, grad_data=False):
    """
    Trainer function for training neural network models.
    
    This function handles the training process of DNN models, including loading pre-trained models,
    setting up training parameters, executing the training loop, and saving the trained model.
    
    Parameters
    ----------
    data : numpy.ndarray
        Training input data.
    target : numpy.ndarray
        Training target values.
    model_path : str
        Path to save or load the model.
    layer : list
        List of integers specifying the number of neurons in each hidden layer.
    learning_rate : float, default=1e-3
        Learning rate for optimization.
    epochs : int, default=30000
        Number of training epochs.
    batch_size : int, default=10
        Batch size for training.
    
    Returns
    -------
    list
        List of training loss values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
    
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
            net = DNN(input_size=np.shape(data)[-1], layer_sizes=layer, output_size=1).double()
            net.load_state_dict(torch.load(model_path))
            print("Loaded pre-trained model (state_dict only).")
    else:
        net = DNN(input_size=np.shape(data)[-1], layer_sizes=layer, output_size=1).double()
        print("Construct model with structure: DNN")
        print(f"Model configuration - Input size: {np.shape(data)[-1]}, "
                    f"Layers: {layer}, " f"Output size: {1}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)
    net.to(device)
    data, target = torch.tensor(data, dtype=torch.float64), torch.tensor(target, dtype=torch.float64)
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)
    data, target = data.to(device), target.to(device)
    dataset = torch.utils.data.TensorDataset(data, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    TrainingLoss = []
    grad_diff_history = []
    hessian_diff_history = []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            if grad_data:
                part_indices = labels[:, 1] == 1    
                actual_grad = labels[part_indices, 2:labels.size()[-1]]
                labels = labels[:, 0].unsqueeze(-1)
                subinputs = inputs[part_indices]
                subinputs.requires_grad_(True)
                suboutputs = net(subinputs)
                outputs_grad = torch.autograd.grad(suboutputs, subinputs,
                                                grad_outputs=torch.ones_like(suboutputs), retain_graph=True)[0]
                loss = custom_loss(outputs, labels, outputs_grad, actual_grad)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if grad_output and epoch % report_interval == 0:
            net.to('cpu')
            result = calculate_derivatives(net)
            net.to(device)
            grad_diff_history.append(result["grad_diff"])
            hessian_diff_history.append(result["hessian_diff"])
            print(f'Epoch {epoch}, Loss: {loss.item()}, first deviation: {result["grad_diff"]}, second deviation: {result["hessian_diff"]}')
            net.to(device)
        elif report and epoch % report_interval == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    net.to('cpu')
    
    model_info = {
        'model_type': 'DNN',
        'input_size': np.shape(data)[-1],
        'layer_sizes': layer,
        'output_size': 1,
        'state_dict': net.state_dict()
    }
    torch.save(model_info, model_path)
    if grad_output:
        model_dir = os.path.dirname(model_path)
        np.save(os.path.join(model_dir, 'grad_diff_history.npy'), np.array(grad_diff_history))
        np.save(os.path.join(model_dir, 'hessian_diff_history.npy'), np.array(hessian_diff_history))

    print(f"Model saved with structure information to {model_path}")
    
    return TrainingLoss
