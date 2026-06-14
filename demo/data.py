import numpy as np
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits

from src.encode.OneHotEncoder import OneHotEncoder
from src.utils import train_test_split


@st.cache_data
def load_data():
    digits = load_digits()
    encoder = OneHotEncoder()
    X = digits.data / digits.data.max()
    y = encoder.encode(digits.target)
    return *train_test_split(X, y, 0.8), encoder


def digit_image(x: np.ndarray) -> Image.Image:
    return Image.fromarray((x.reshape(8, 8) * 255).astype(np.uint8)).resize((64, 64), Image.NEAREST)
