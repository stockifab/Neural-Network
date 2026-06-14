from typing import Literal

import numpy as np

from src.network import Layer
from src.network import Connection

type Loss = Literal["binary_crossentropy", "categorical_crossentropy"]


def cost_wrt_z(a: np.ndarray, y_true: np.ndarray):
    return -(y_true - a)


class Network:
    def __init__(self, layers: list[Layer], loss: Loss = "binary_crossentropy"):
        self.loss = loss
        self.__build_pipeline(layers)

    def __build_pipeline(self, layers: list[Layer]):
        self.pipeline: list[Connection | Layer] = []
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            self.pipeline.append(layer)
            self.pipeline.append(Connection(layer.n_nodes, next_layer.n_nodes))
        self.pipeline.append(layers[-1])

    def forward(self, inp: np.ndarray):
        last: np.ndarray = inp
        for item in self.pipeline:
            last = item.run(last)
        return last

    def cost(self, inp: np.ndarray, y_true: np.ndarray):
        a = self.forward(inp)
        if self.loss == "binary_crossentropy":
            return np.sum(-(y_true * np.log(a) + (1 - y_true) * np.log(1 - a)))
        elif self.loss == "categorical_crossentropy":
            return np.sum(-(y_true * np.log(a)))

    def epoch(self, inp: np.ndarray, y_true, learning_rate: float = 0.01):
        y_true = np.array(y_true)

        activations: list[np.ndarray] = [inp]
        last: np.ndarray = inp
        for item in self.pipeline:
            last = item.run(last)
            if isinstance(item, Layer):
                activations.append(last)

        a_idx = len(activations) - 1
        err = None

        for item in reversed(self.pipeline):
            if isinstance(item, Layer):
                a = activations[a_idx]

                if err is None:
                    err = cost_wrt_z(a, y_true)
                else:
                    if item.activation == "sigmoid":
                        err = err * a * (1 - a)
                    elif item.activation == "softmax":
                        dot = np.sum(err * a, axis=1, keepdims=True)
                        err = a * (err - dot)

                item.biases -= learning_rate * err.mean(axis=0)

                a_idx -= 1

            elif isinstance(item, Connection):
                a_prev = activations[a_idx]

                weight_grad = a_prev.T @ err / len(inp)

                err = err @ item.weights.T

                item.weights -= learning_rate * weight_grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 5_000,
        learning_rate: float = 0.01,
        batch_size: float = 0.2,
        verbose=True,
        epoch_msg=100,
    ):
        for epoch in range(1, epochs + 1):
            indices = np.random.choice(len(X), int(len(X) * batch_size), replace=False)

            self.epoch(X[indices], y[indices], learning_rate)
            if verbose and epoch % epoch_msg == 0:
                cost = self.cost(X, y)
                print(f"[{epoch:{len(str(epochs))}} / {epochs}] Cost: {cost}")
