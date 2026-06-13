import numpy as np


def scalar_sigmoid(z):
    return 1 / (1 + np.e ** (-z))


sigmoid = np.vectorize(scalar_sigmoid)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return shifted / np.sum(shifted, axis=-1, keepdims=True)
