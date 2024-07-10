"""Provides MLPArchive."""
import numpy as np
import torch
from torch import nn


class MLPArchive(nn.Module):
    """MLP archive.

    Feedforward network with identical activations on every layer. Takes in a
    feature and outputs a solution. There is no activation on the last layer.

    Some methods return self so that you can do archive = MLPArchive().method()

    Args:
        layer_specs: List of tuples of (in_shape, out_shape, bias (optional))
            for linear layers.
        activation: Activation layer class, e.g. nn.Tanh
    """

    def __init__(self, layer_specs, activation):
        super().__init__()

        layers = []
        for i, shape in enumerate(layer_specs):
            layers.append(
                nn.Linear(shape[0],
                          shape[1],
                          bias=shape[2] if len(shape) == 3 else True))
            if i != len(layer_specs) - 1:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        """Computes solutions for a batch of features."""
        return self.model(features)

    def initialize(self, func):
        """Initializes weights for Linear layers with func.

        func usually comes from nn.init
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Biases init to zero.

        self.apply(init_weights)

        return self

    def n_params(self):
        """Number of parameters in the model."""
        return sum(np.prod(p.shape) for p in self.parameters())

    def serialize(self):
        """Returns 1D array with all parameters in the model."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.model.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self

    def gradient(self):
        """Returns 1D array with gradient of all parameters in the model."""
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()])
