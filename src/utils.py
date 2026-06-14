import numpy as np


def train_test_split(
    X: np.ndarray, y: np.ndarray, train_size: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices = int(len(X) * train_size)

    return X[:train_indices], X[train_indices:], y[:train_indices], y[train_indices:]
