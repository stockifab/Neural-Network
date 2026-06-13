# Neural Network from Scratch

A neural network implemented in pure NumPy, trained on the sklearn digits dataset (handwritten digit classification).

## Structure

```
src/network/
  Layer.py       – layer with bias and activation (input / sigmoid / softmax)
  Connection.py  – weight matrix between layers
  Network.py     – forward pass, backprop, cost
  activation.py  – sigmoid and softmax implementations
main.py          – trains a 64→50→30→10 network on the digits dataset
```

## Usage

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python main.py
```

Trains for 10 000 epochs with mini-batches (10 % of data), prints cost every 100 epochs, then shows a loss curve and final accuracy.
