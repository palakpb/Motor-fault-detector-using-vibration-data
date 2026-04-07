import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

st.set_page_config(page_title="Motor Fault Detection", layout="wide")

st.title("Motor Fault Detection using Vibration Data")
st.markdown("A machine learning project to detect bearing faults in industrial motors using the CWRU Bearing Dataset.")

df = pd.read_csv("model_results.csv")

st.header("Model Comparison")
st.markdown("Two models were trained and evaluated on the same dataset. Random Forest outperformed SVM significantly.")
st.dataframe(df, use_container_width=True)

st.subheader("Accuracy Chart")
chart_df = df.set_index("Model")
st.bar_chart(chart_df)

st.header("Why Random Forest Performed Better")
st.write("Random Forest is an ensemble method that builds multiple decision trees and combines their outputs. This makes it robust to noise in vibration signals. SVM works well for smaller datasets but struggles when the feature space is complex and the data has overlapping fault patterns.")

st.header("Confusion Matrix")
if os.path.exists("confusion_matrices.png"):
    img1 = Image.open("confusion_matrices.png")
    st.image(img1, use_container_width=True)
    st.markdown("The confusion matrix shows how well the model classified each fault type. A near-perfect diagonal means very few misclassifications.")
else:
    st.warning("confusion_matrices.png not found")

st.header("FFT Spectra")
if os.path.exists("fft_spectra.png"):
    img2 = Image.open("fft_spectra.png")
    st.image(img2, use_container_width=True)
    st.markdown("FFT (Fast Fourier Transform) converts raw vibration signals from time domain to frequency domain. Each fault type produces a unique frequency signature which the model uses to classify the condition.")
else:
    st.warning("fft_spectra.png not found")

st.header("Fault Categories")
fault_data = {
    "Fault Type": ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"],
    "Description": [
        "Bearing is healthy with no defect",
        "Defect present on the rolling ball element",
        "Defect present on the inner ring of the bearing",
        "Defect present on the outer ring of the bearing"
    ],
    "Risk Level": ["None", "Medium", "High", "High"]
}
st.table(pd.DataFrame(fault_data))

st.header("Project Workflow")
st.markdown("""
1. Raw vibration signals collected from CWRU Bearing Dataset
2. FFT applied to convert signals to frequency domain
3. Statistical features extracted from FFT output
4. Features scaled using Standard Scaler
5. Random Forest and SVM models trained and compared
6. Best model saved and deployed in this web app
""")

st.header("About")
st.write("This project was built as part of an academic machine learning project to explore real world applications of signal processing and classification. The CWRU Bearing Dataset is a standard benchmark dataset used in predictive maintenance research. The goal was to build a system that can automatically detect the type of fault in a motor bearing using only vibration data, without any manual inspection.")
st.write("Tools used: Python, Scikit-learn, NumPy, Pandas, Streamlit")
st.write("Dataset: Case Western Reserve University Bearing Dataset")
