from typing import Callable

import numpy as np
from src.network.activation import sigmoid


class Layer:
    def __init__(self, n_nodes: int, activation: Callable[[float], float] = sigmoid):
        self.n_nodes = n_nodes
        self.activation = np.vectorize(activation)

        self.biases = np.zeros(n_nodes)

    def run(self, inp: np.ndarray):
        return self.activation(inp + self.biases.T)
