import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# --- App title ---
st.title("🌍 Economic Well-Being Predictor")

# --- Upload input CSV ---
uploaded_file = st.file_uploader("Upload Test CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Input Data")
    st.write(data.head())

    # --- Preprocessing ---
    # Frequency encode as done in training
    data['country'] = data['country'].astype(str)
    data['urban_or_rural'] = data['urban_or_rural'].astype(str)

    for col in ['country', 'urban_or_rural']:
        freq = data[col].value_counts() / len(data)
        data[col] = data[col].map(freq)

    # Drop unused columns
    features = [col for col in data.columns if col not in ['ID']]
    input_data = data[features]

    # --- Load trained model ---
    model = joblib.load("model.pkl")

    # --- Predict ---
    predictions = model.predict(input_data)
    data['Predicted Target'] = predictions

    st.subheader("📈 Predictions")
    st.write(data[['ID', 'Predicted Target']].head())

    # --- Downloadable output ---
    csv = data[['ID', 'Predicted Target']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction CSV", data=csv, file_name="submission.csv", mime="text/csv")