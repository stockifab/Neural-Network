import numpy as np
from src.encode.OneHotEncoder import OneHotEncoder
from src.network import Layer
from src.network import Network
from sklearn.datasets import load_digits

from src.utils import train_test_split

digits = load_digits()

encoder = OneHotEncoder()

X = digits.data / digits.data.max()
y = encoder.encode(digits.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

np.random.seed(42)

network = Network(
    [
        Layer(64, activation="input"),
        Layer(50),
        Layer(10, activation="softmax"),
    ],
    loss="categorical_crossentropy",
)

network.fit(X_train, y_train, epochs=600, learning_rate=0.9, batch_size=0.1)

print("Final Cost:", network.cost(X_train, y_train))

uq = np.unique_counts(encoder.decode(network.forward(X_test)) == encoder.decode(y_test))
tf_count = dict(zip(uq.values, uq.counts))
print("Accuracy: ", tf_count[True] / len(X_test))
