"""Small display helpers shared by the DIP Lab topic modules."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocess as pp


def fit(img, width=880):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA) \
        if img.shape[1] > width else img


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)


def gallery(images, per_row=3):
    """Captioned images in a grid. `images` is a dict caption -> image."""
    items = list(images.items())
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (caption, img) in zip(cols, items[i:i + per_row]):
            col.image(rgb(img), caption=caption, width="stretch")


def hist_frame(images, fn=pp.histogram, scale=100.0):
    return pd.concat([pd.DataFrame({"level": np.arange(256), "value": fn(img) * scale, "image": name})
                      for name, img in images.items()], ignore_index=True)


def entropy(img):
    h = pp.histogram(img)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def contrast(img):
    return float(pp.luminance(img).std())


def measure(images, extra=None):
    rows = []
    for name, img in images.items():
        row = {"Image": name, "Mean": round(pp.mean_brightness(img), 1), "Contrast (σ)": round(contrast(img), 1),
               "Entropy (bits)": round(entropy(img), 2), "Sharpness (Lap. var)": round(pp.blur_score(img), 1)}
        if extra:
            row.update(extra(name, img))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def note(text):
    st.markdown(f"<div class='av-note'>{text}</div>", unsafe_allow_html=True)
