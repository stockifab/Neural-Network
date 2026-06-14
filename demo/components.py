import numpy as np
import streamlit as st
import plotly.graph_objects as go

from data import digit_image


def loss_chart(x: list[int], y: list[float], x_end: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#636EFA", width=2.5)))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=280,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(range=[0, x_end]),
        yaxis=dict(rangemode="tozero"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def digit_row(
    images: np.ndarray,
    preds: list[int] | None = None,
    trues: list[int] | None = None,
) -> None:
    for i, col in enumerate(st.columns(len(images))):
        col.image(digit_image(images[i]), width="stretch")
        if preds is None:
            continue
        correct = preds[i] == trues[i]
        color = "#2ca02c" if correct else "#d62728"
        col.markdown(
            f"<div style='text-align:center;margin-top:-4px;line-height:1.3'>"
            f"<span style='font-size:1rem;font-weight:700;color:{color}'>{preds[i]}</span>"
            f"<span style='font-size:0.75rem;opacity:0.45'> / {trues[i]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
