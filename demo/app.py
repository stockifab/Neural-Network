import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
# when running: streamlit run demo/app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from data import load_data
from views import welcome, training_view, results_view

# ── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Neural Network Demo", layout="wide")
st.title("Neural Network from Scratch")
st.caption("A handwritten digit classifier trained with pure NumPy — no PyTorch, no TensorFlow.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Architecture")
n_hidden = st.sidebar.slider("Hidden layers", 1, 4, 1)
defaults = [50, 30, 20, 15]
hidden_sizes = [st.sidebar.slider(f"Layer {i + 1} neurons", 5, 128, defaults[i]) for i in range(n_hidden)]

st.sidebar.header("Training")
lr = st.sidebar.number_input("Learning rate", 0.01, 10.0, 0.9, step=0.1, format="%.2f")
epochs = st.sidebar.slider("Epochs", 100, 10_000, 600, step=100)
batch_size = st.sidebar.slider("Batch fraction", 0.05, 1.0, 0.1, step=0.05)
seed = st.sidebar.number_input("Random seed", 0, 9_999, 42)

train_btn = st.sidebar.button("Train Network", type="primary")

# ── Data ─────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test, encoder = load_data()

# ── Routing ───────────────────────────────────────────────────────────────────

if train_btn:
    training_view(hidden_sizes, lr, epochs, batch_size, int(seed), X_train, y_train)

if "network" in st.session_state:
    results_view(X_train, X_test, y_train, y_test, encoder)
elif not train_btn:
    welcome(X_test, y_test, len(X_train), len(X_test))
