import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP ---
st.set_page_config(page_title="Glucose Prediction AI", page_icon="🩸")
st.title("🩸 Glucose Prediction AI")

@st.cache_resource
def load_resources():
    # A. Load the Model from your PKL file
    # The inspection showed this file contains the Keras model
    try:
        with open('glucose_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None, None

    # B. Manually Recreate the Scaler
    # We use the exact values found in your 'model.h' file
    scaler = StandardScaler()
    
    # Values extracted from your model.h
    # Order: heart_rate, skin_temp, gsr, age, weight
    scaler.mean_ = np.array([85.227193, 36.508328, 1.498241, 42.621742, 79.629291])
    scaler.scale_ = np.array([14.571841, 0.574858, 0.579961, 18.928609, 18.205225])
    
    # These lines tell the scaler it is "ready" to use
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_samples_seen_ = 100 
    
    return model, scaler

model, scaler = load_resources()

if model is not None:
    st.success("✅ Model and Scaler loaded successfully!")

    # --- 2. USER INPUTS ---
    st.markdown("### Enter Patient Vitals")
    st.info("Input Order: Heart Rate → Skin Temp → GSR → Age → Weight")
    
    col1, col2 = st.columns(2)

    with col1:
        heart_rate = st.number_input("Heart Rate (BPM)", value=85.0)
        skin_temp = st.number_input("Skin Temperature (°C)", value=36.5)
        gsr = st.number_input("GSR (Galvanic Skin Response)", value=1.5)

    with col2:
        age = st.number_input("Age (Years)", value=42.0)
        weight = st.number_input("Weight (kg)", value=79.0)

    # --- 3. PREDICTION ---
    if st.button("Predict Glucose Level"):
        # 1. Arrange inputs in the exact order the model expects
        input_data = np.array([[heart_rate, skin_temp, gsr, age, weight]])
        
        # 2. Scale the data using our manual scaler
        scaled_data = scaler.transform(input_data)
        
        # 3. Predict
        try:
            prediction = model.predict(scaled_data)
            
            # Extract the single value
            glucose_value = prediction[0][0]
            
            st.divider()
            st.metric("Predicted Glucose", f"{glucose_value:.2f} mg/dL")
            
            # Simple interpretation
            if glucose_value < 70:
                st.warning("⚠️ Low Glucose (Hypoglycemia)")
            elif glucose_value > 140:
                st.error("⚠️ High Glucose (Hyperglycemia)")
            else:
                st.success("✅ Normal Range")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")