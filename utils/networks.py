import torch
import torch.nn as nn

class DNN(nn.Module):
    """Deep Neural Network model for function approximation.
    
    A fully connected neural network with tanh activation functions
    and linear output layer.
    """
    def __init__(self, input_size, layer_sizes, output_size):
        """Initialize the DNN model with specified architecture.
        
        Args:
            input_size: Number of input features.
            layer_sizes: List of integers specifying the number of neurons in each hidden layer.
            output_size: Number of output features.
        """
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, x):
        """Forward pass through the neural network.
        
        Args:
            x: Input tensor to process.
            
        Returns:
            Output tensor after passing through the network.
        """
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def custom_loss(outputs, labels, outputs_grad, actual_grad, k=0.002):
    base_loss = torch.nn.functional.mse_loss(outputs, labels)
    if len(actual_grad) == 0:
        return base_loss
    grad_diff = torch.nn.functional.mse_loss(outputs_grad, actual_grad)
    total_loss = base_loss + k * grad_diff

    return total_loss