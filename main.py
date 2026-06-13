import numpy as np
from src.network import Layer
from src.network import Network
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

full_digits = load_digits()

train_size = int(len(full_digits.data) * 0.8)
train_digits = full_digits.data[train_size:]
test_digits = full_digits.data[:train_size]
train_digits_target = full_digits.target[train_size:]
test_digits_target = full_digits.target[:train_size]

train_data = train_digits.data / train_digits.max()
test_data = test_digits.data / test_digits.max()

train_target = np.array(
    [np.array([float(x == i) for i in range(10)]) for x in train_digits_target]
)
test_target = np.array(
    [np.array([float(x == i) for i in range(10)]) for x in train_digits_target]
)

np.random.seed(42)

network = Network(
    [
        Layer(64, activation="input"),
        Layer(50),
        Layer(30),
        Layer(10, activation="softmax"),
    ],
    loss="categorical_crossentropy",
)

cost_per_epoch = pd.DataFrame(columns=["epoch", "cost"])

EPOCHS = 10000
for epoch in range(EPOCHS):
    indices = np.random.choice(len(train_data), len(train_data) // 10, replace=False)

    network.fit(train_data[indices], train_target[indices], 0.1)
    if epoch % 100 == 0:
        cost = network.cost(train_data, train_target)
        cost_per_epoch.loc[len(cost_per_epoch)] = (epoch, cost)
        print(f"[Epoch {epoch}/{EPOCHS}] ", cost)


print("Result: ", network.run(train_data[0]))
print("Cost:   ", network.cost(train_data, train_target))

uq = np.unique_counts(network.run(test_data).argmax(axis=1) == test_digits_target)
tf_count = dict(zip(uq.values, uq.counts))
print("Accuracy: ", tf_count[True] / len(test_data))

sns.set_style("whitegrid")
sns.set_context("notebook")
sns.lineplot(cost_per_epoch, x="epoch", y="cost")
plt.title("Cost at each epoch")
plt.show()
