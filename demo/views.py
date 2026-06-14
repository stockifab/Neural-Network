import numpy as np
import streamlit as st

from src.encode.OneHotEncoder import OneHotEncoder
from src.network import Layer, Network

from components import digit_row, loss_chart


def welcome(X_test: np.ndarray, y_test: np.ndarray, n_train: int, n_test: int) -> None:
    st.markdown(f"""
### Welcome

This demo trains a fully-connected neural network built from scratch with NumPy on the
[sklearn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) —
1,797 grayscale 8×8 images of handwritten digits (classes 0–9), split into
**{n_train} training** and **{n_test} test** samples.

Configure the network architecture and training parameters in the sidebar, then click
**Train Network** to watch the loss converge in real time.

---
**Dataset preview**
""")
    idx = np.random.default_rng(7).choice(len(X_test), 10, replace=False)
    digit_row(X_test[idx])


def training_view(
    hidden_sizes: list[int],
    lr: float,
    epochs: int,
    batch_size: float,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    np.random.seed(seed)

    layers = [Layer(64, activation="input"), *[Layer(s) for s in hidden_sizes], Layer(10, activation="softmax")]
    network = Network(layers, loss="categorical_crossentropy")

    costs: list[float] = []
    logged_epochs: list[int] = []
    log_every = max(1, epochs // 50)

    progress = st.progress(0)
    chart_slot = st.empty()

    for epoch in range(1, epochs + 1):
        idx = np.random.choice(len(X_train), int(len(X_train) * batch_size), replace=False)
        network.epoch(X_train[idx], y_train[idx], lr)

        if epoch % log_every == 0 or epoch == epochs:
            costs.append(network.cost(X_train, y_train))
            logged_epochs.append(epoch)
            chart_slot.plotly_chart(
                loss_chart(logged_epochs, costs, epochs),
                width="stretch",
                key=f"live_{epoch}",
            )

        progress.progress(epoch / epochs, text=f"Epoch {epoch} / {epochs}")

    progress.empty()
    chart_slot.empty()
    st.session_state.update(network=network, costs=costs, logged_epochs=logged_epochs)


def results_view(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: OneHotEncoder,
) -> None:
    network: Network = st.session_state["network"]
    costs: list[float] = st.session_state["costs"]
    logged_epochs: list[int] = st.session_state["logged_epochs"]

    train_acc = np.mean(encoder.decode(network.forward(X_train)) == encoder.decode(y_train))
    test_acc = np.mean(encoder.decode(network.forward(X_test)) == encoder.decode(y_test))

    col_chart, col_metrics = st.columns([3, 1])
    with col_chart:
        st.subheader("Training loss")
        st.plotly_chart(loss_chart(logged_epochs, costs, logged_epochs[-1]), width="stretch", key="final")
    with col_metrics:
        st.subheader("Metrics")
        st.metric("Train accuracy", f"{train_acc:.1%}")
        st.metric("Test accuracy", f"{test_acc:.1%}")
        st.metric("Final loss", f"{costs[-1]:.4f}")

    st.divider()
    st.subheader("Sample predictions")
    st.caption("Predicted (large) / true (small) — green = correct, red = wrong")

    indices = np.random.default_rng(0).choice(len(X_test), 10, replace=False)
    preds = encoder.decode(network.forward(X_test[indices])).astype(int)
    trues = encoder.decode(y_test[indices]).astype(int)
    digit_row(X_test[indices], preds.tolist(), trues.tolist())
