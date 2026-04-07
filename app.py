import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Motor Fault Detection", layout="wide")

st.title("Motor Fault Detection System")
st.markdown("Random Forest Classifier on CWRU Bearing Dataset")

df = pd.read_csv("model_results.csv")

st.header("Model Performance")
st.dataframe(df, use_container_width=True)

st.header("Confusion Matrix")
if os.path.exists("confusion_matrices.png"):
    img1 = Image.open("confusion_matrices.png")
    st.image(img1, use_column_width=True)
else:
    st.warning("confusion_matrices.png not found")

st.header("FFT Spectra")
if os.path.exists("fft_spectra.png"):
    img2 = Image.open("fft_spectra.png")
    st.image(img2, use_column_width=True)
else:
    st.warning("fft_spectra.png not found")

st.header("About")
st.write("This project uses the CWRU Bearing Dataset to classify motor faults using vibration signal features extracted via FFT. A Random Forest model was trained to detect Normal, Ball Fault, Inner Race Fault, and Outer Race Fault conditions.")
