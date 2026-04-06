import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from src.predict import predict_system

# ============================
# LOAD MODEL (for importance)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))

# ============================
# UI TITLE
# ============================
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("ISRO Predictive Maintenance Dashboard")

# ============================
# SIDEBAR INPUT
# ============================
st.sidebar.header("Input Parameters")

Type = st.sidebar.selectbox("Machine Type", [0, 1, 2])
air_temp = st.sidebar.slider("Air Temperature", 290, 330, 300)
process_temp = st.sidebar.slider("Process Temperature", 300, 340, 310)
speed = st.sidebar.slider("Rotational Speed", 1200, 2000, 1500)
torque = st.sidebar.slider("Torque", 20, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear", 0, 30, 5)

features = [Type, air_temp, process_temp, speed, torque, tool_wear]

# ============================
# MAIN LAYOUT
# ============================
col1, col2 = st.columns(2)

# ============================
# PREDICTION PANEL
# ============================
with col1:
    st.subheader("Prediction Output")

    if st.button("Run Prediction"):
        prob, decision = predict_system(features)

        # Probability metric
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Progress bar
        st.progress(int(prob * 100))

        # Risk Indicator
        if prob < 0.3:
            st.success("SAFE")
        elif prob < 0.7:
            st.warning("WARNING")
        else:
            st.error("CRITICAL")

        # RL Decision
        st.subheader("Recommended Action")
        st.write(decision)

        # Explanation
        st.subheader("Explanation")
        if prob < 0.3:
            st.write("System is stable. No maintenance required.")
        elif prob < 0.7:
            st.write("Moderate risk detected. Preventive maintenance recommended.")
        else:
            st.write("Critical condition. Immediate shutdown required.")

        # Alert
        if prob > 0.7:
            st.error("ALERT: High Failure Risk!")

# ============================
# VISUALIZATION PANEL
# ============================
with col2:
    st.subheader("System Insights")

    # Feature Importance
    st.write("### Feature Importance")

    try:
        importance = model.feature_importances_
        features_names = ['Type', 'Air Temp', 'Process Temp', 'Speed', 'Torque', 'Tool Wear']

        fig, ax = plt.subplots()
        ax.barh(features_names, importance)
        ax.set_title("Feature Importance")

        st.pyplot(fig)
    except:
        st.write("Feature importance not available.")

    # Sensor Simulation
    st.write("### Sensor Trend Simulation")

    data = np.random.normal(air_temp, 2, 50)
    st.line_chart(data)

# ============================
# INPUT SUMMARY
# ============================
st.sidebar.subheader("Input Summary")
st.sidebar.write(f"Type: {Type}")
st.sidebar.write(f"Air Temp: {air_temp}")
st.sidebar.write(f"Process Temp: {process_temp}")
st.sidebar.write(f"Speed: {speed}")
st.sidebar.write(f"Torque: {torque}")
st.sidebar.write(f"Tool Wear: {tool_wear}")