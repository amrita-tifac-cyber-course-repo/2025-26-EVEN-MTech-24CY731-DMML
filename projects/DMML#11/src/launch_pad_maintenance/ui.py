import streamlit as st
from src.predict import predict_system

st.title("🚀 Predictive Maintenance System")

Type = st.selectbox("Type", [0, 1, 2])
air_temp = st.number_input("Air Temperature", value=300.0)
process_temp = st.number_input("Process Temperature", value=310.0)
speed = st.number_input("Rotational Speed", value=1500.0)
torque = st.number_input("Torque", value=40.0)
tool_wear = st.number_input("Tool Wear", value=5.0)

if st.button("Predict"):
    features = [Type, air_temp, process_temp, speed, torque, tool_wear]
    
    prob, decision = predict_system(features)
    
    st.success(f"Failure Probability: {prob:.2f}")
    st.warning(f"Recommended Action: {decision}")