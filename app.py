import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Motor Fault Detection",
    page_icon="⚙️",
    layout="wide"
)

# Title
st.title("⚙️ Motor Fault Detection System")
st.markdown("### Using Random Forest on CWRU Bearing Dataset")
st.markdown("---")

# Load model results
@st.cache_data
def load_results():
    return pd.read_csv("model_results.csv")

df = load_results()

# --- Section 1: Model Performance ---
st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

# Try to show key metrics if columns exist
try:
    accuracy = df[df.columns[df.columns.str.contains('accuracy', case=False)][0]].values[0]
    col1.metric("✅ Accuracy", f"{float(accuracy)*100:.2f}%")
except:
    pass

try:
    precision = df[df.columns[df.columns.str.contains('precision', case=False)][0]].values[0]
    col2.metric("🎯 Precision", f"{float(precision)*100:.2f}%")
except:
    pass

try:
    recall = df[df.columns[df.columns.str.contains('recall', case=False)][0]].values[0]
    col3.metric("🔁 Recall", f"{float(recall)*100:.2f}%")
except:
    pass

st.markdown("#### Full Results Table")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# --- Section 2: Confusion Matrix ---
st.header("🔢 Confusion Matrix")
try:
    img1 = Image.open("confusion_matrices.png")
    st.image(img1, use_column_width=True)
except:
    st.warning("confusion_matrices.png not found")

st.markdown("---")

# --- Section 3: FFT Spectra ---
st.header("📈 FFT Spectra")
try:
    img2 = Image.open("fft_spectra.png")
    st.image(img2, use_column_width=True)
except:
    st.warning("fft_spectra.png not found")

st.markdown("---")

# --- Section 4: About ---
st.header("ℹ️ About This Project")
st.markdown("""
- **Dataset:** CWRU Bearing Dataset
- **Model:** Random Forest Classifier
- **Goal:** Detect motor faults from vibration signal features
- **Classes:** Normal, Ball Fault, Inner Race Fault, Outer Race Fault
""")