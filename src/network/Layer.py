from typing import Literal

import numpy as np
from src.network.activation import sigmoid, softmax

type Activation = Literal["input", "sigmoid", "softmax"]


class Layer:
    def __init__(self, n_nodes: int, activation: Activation = "sigmoid"):
        self.n_nodes = n_nodes
        self.activation: Activation = activation

        self.biases = np.zeros(n_nodes)

    def run(self, inp: np.ndarray):
        z = inp + self.biases.T
        if self.activation == "input":
            return z
        elif self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "softmax":
            return softmax(z)
