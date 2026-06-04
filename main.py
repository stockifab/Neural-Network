from src.network import Layer
from src.network import Network
from src.generate import generate
import numpy as np

np.random.seed(42)

data, target = generate()
data = data / 100.0
target = [(float(t[0] == 0), float(t[0] == 1)) for t in target]

network = Network(
    [
        Layer(2, activation=lambda x: x),
        Layer(5),
        Layer(5),
        Layer(2),
    ]
)

print("Result: ", network.run(data[0]))
print("Cost:   ", network.cost(data, target))

EPOCHS = 6800
lr = 0.1
for epoch in range(EPOCHS):
    if epoch > 1000:
        lr = 0.5
    if epoch > 5000:
        lr = 0.005
    network.fit(data, target, 0.05)
    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{EPOCHS}] ", network.cost(data, target))


print("Result: ", network.run(data[0]))
print("Cost:   ", network.cost(data, target))
