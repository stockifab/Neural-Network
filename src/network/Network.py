import numpy as np

from src.network import Layer
from src.network import Connection


def cost_wrt_z(a: np.ndarray, y_true: np.ndarray):
    return -2 * (y_true - a) * a * (1 - a)


class Network:
    def __init__(self, layers: list[Layer]):
        self.__build_pipeline(layers)

    def __build_pipeline(self, layers: list[Layer]):
        self.pipeline: list[Connection | Layer] = []
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            self.pipeline.append(layer)
            self.pipeline.append(Connection(layer.n_nodes, next_layer.n_nodes))
        self.pipeline.append(layers[-1])

    def run(self, inp: np.ndarray):
        last: np.ndarray = inp
        for item in self.pipeline:
            last = item.run(last)
        return last

    def cost(self, inp: np.ndarray, y_true: np.ndarray):
        y_pred = self.run(inp)
        return np.sum((y_pred - y_true) ** 2)

    def fit(self, inp: np.ndarray, y_true, learning_rate: float = 0.01):
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
                    err = err * a * (1 - a)

                item.biases -= learning_rate * err.mean(axis=0)

                a_idx -= 1

            elif isinstance(item, Connection):
                a_prev = activations[a_idx]

                weight_grad = a_prev.T @ err / len(inp)

                err = err @ item.weights.T

                item.weights -= learning_rate * weight_grad
