
import torch
import torch.nn as nn

from typing import List, Literal


class MonotoneLayer(nn.Module):
    """
            Monotone Layer

            parameters
            ----------
            in_dim: int
                input dimension
            out_dim: int
                output dimension
        """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, out_dim, in_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1, out_dim), requires_grad=True)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.activation((torch.square(self.weights) @ x).squeeze(dim=-1) + self.bias)


class NormLayer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.rand(1, dim), requires_grad=True)
        self.scale = nn.Parameter(torch.rand(1, dim), requires_grad=True)

    def forward(self, x):
        return torch.square(self.scale) * x - self.bias


class MonoNetwork(nn.Module):
    """
            Feedforward Neural Network approximating learning (multivariate) functions, such that for each input
            variable each network output is either nondecreasing or nonincreasing depending on configuration

            parameters
            ----------
            architecture: List[int]
                list of number of neurons per layer starting with the first hidden layer
            input_monotonicity: List[Literal[-1, 1]]
                List of monotonicity-type indicators. The length of the list corresponds to the input dimension
                of the network. -1 indicates, that the network outputs shall be nonincreasing in the input variable, 1 that
                they shall be nondecreasing.

        """

    def __init__(self, hidden_layers: List[int], input_monotonicity: List[Literal[-1, 1]]):
        super().__init__()
        self.register_buffer('monotonicity', torch.tensor([input_monotonicity]))
        layers = [len(input_monotonicity)] + hidden_layers
        self.layers = nn.ModuleList([NormLayer(len(input_monotonicity))]
                                     + [MonotoneLayer(in_dim, out_dim) for in_dim, out_dim in zip(layers, layers[1:])]
                                     + [NormLayer(hidden_layers[-1])])

    def forward(self, x):
        x_mono = self.monotonicity * x
        for layer in self.layers:
            x_mono = layer(x_mono)
        return x_mono
