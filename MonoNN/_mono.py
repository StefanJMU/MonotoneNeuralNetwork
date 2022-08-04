
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
        self.shift = nn.Parameter(torch.rand(1, out_dim, in_dim), requires_grad=True)
        self.weights = nn.Parameter(torch.rand(1, out_dim, in_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1, out_dim), requires_grad=True)
        self.input_activation = nn.Tanh()
        self.output_activation = nn.Tanh()  # account for the contingency, that these activations might not be the same
        self.out_dim = out_dim

    def forward(self, x):
        # -> (batch, out_dim, in_dim)
        x_expanded = torch.repeat_interleave(torch.unsqueeze(x, 1), repeats=self.out_dim, dim=1)
        x_shifted = x_expanded - self.shift
        x_activated = self.input_activation(x_shifted)

        weights_squared = torch.square(self.weights)
        x_weighted = weights_squared * x_activated

        # -> (batch, out_dim)
        x = torch.sum(x_weighted, dim=-1)
        return self.output_activation(x)


class MonoNN(nn.Module):

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

    def __init__(self, architecture: List[int], input_monotonicity: List[Literal[-1, 1]]):
        super().__init__()
        self.register_buffer('monotonicity', torch.tensor([input_monotonicity], dtype=torch.float32))
        layers = [len(input_monotonicity)] + architecture
        self.layers = nn.ModuleList([
                MonotoneLayer(in_dim, out_dim) for in_dim, out_dim in zip(layers, layers[1:])
        ])

    def forward(self, x):
        x_mono = self.monotonicity * x
        for layer in self.layers:
            x_mono = layer(x_mono)
        return x_mono