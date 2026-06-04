import numpy as np


class Connection:
    def __init__(self, n_from: int, n_to: int):
        self.weights = np.random.randn(n_from, n_to) * np.sqrt(1.0 / n_from)

    def run(self, inp: np.ndarray):
        return inp @ self.weights
