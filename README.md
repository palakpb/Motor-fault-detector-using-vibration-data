# Motor-fault-detector-using-vibration-data
ML-powered motor fault detection system using Random Forest on CWRU Bearing Dataset — deployed as an interactive web app
# ⚙️ Motor Fault Detection System

A machine learning project that detects bearing faults in industrial 
motors using vibration signal analysis.

## 🎯 What This Project Does
- Analyzes vibration signals from the CWRU Bearing Dataset
- Extracts frequency domain features using FFT
- Classifies motor condition into 4 fault types using Random Forest
- Deployed as an interactive web dashboard using Streamlit

 Fault Categories Detected
| Fault Type | Description |
|------------|-------------|
| Normal | Healthy bearing, no fault |
| Ball Fault | Defect on the rolling ball |
| Inner Race Fault | Defect on the inner ring |
| Outer Race Fault | Defect on the outer ring |

 Tech Stack
- **Python** — core language
- **Scikit-learn** — Random Forest model
- **NumPy / Pandas** — data processing
- **FFT (Fast Fourier Transform)** — signal feature extraction
- **Streamlit** — web app deployment

 Results
- Trained and evaluated on the CWRU Bearing Dataset
- Model performance metrics visible in the live app

🚀 Live Demo
👉 [Click here to view the app](YOUR_STREAMLIT_LINK_HERE)
