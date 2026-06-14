import numpy as np


class OneHotEncoder:
    def __init__(self):
        self.label_to_index = None

    def encode(self, y: np.ndarray):
        y = np.asarray(y)
        unique = np.unique(y)

        self.label_to_index = {label: idx for idx, label in enumerate(unique)}
        one_hot = np.zeros((y.shape[0], unique.shape[0]), dtype=int)
        for i, val in enumerate(y):
            one_hot[i, self.label_to_index[val]] = 1
        return one_hot

    def decode(self, y_encoded: np.ndarray):
        if not self.label_to_index:
            raise Exception("label to index map missing, call encode() before decode()")

        index_to_label = {val: key for key, val in self.label_to_index.items()}

        y_encoded = np.asarray(y_encoded)
        y = np.zeros(y_encoded.shape[0])

        for i, vals in enumerate(y_encoded):
            max_idx = np.argmax(vals)
            y[i] = index_to_label[max_idx]

        return y
