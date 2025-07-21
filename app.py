import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
encoder = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Customer Sales Prediction", page_icon="📊", layout="centered")
st.title("📊 Customer Sales Prediction App")

st.markdown("Enter customer details below to predict sales behavior using a trained machine learning model.")

with st.form("prediction_form"):
    st.subheader("🧍 Customer Details")

    age = st.slider("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", encoder['workclass'].classes_)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 150000)
    education_num = st.slider("Education Number", 1, 20, 10)
    marital_status = st.selectbox("Marital Status", encoder['marital-status'].classes_)
    occupation = st.selectbox("Occupation", encoder['occupation'].classes_)
    relationship = st.selectbox("Relationship", encoder['relationship'].classes_)
    race = st.selectbox("Race", encoder['race'].classes_)
    sex = st.selectbox("Sex", encoder['sex'].classes_)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
    hours_per_week = st.slider("Hours Per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", encoder['native-country'].classes_)

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        try:
            input_data = {
                'age': age,
                'workclass': encoder['workclass'].transform([workclass])[0],
                'fnlwgt': fnlwgt,
                'education-num': education_num,
                'marital-status': encoder['marital-status'].transform([marital_status])[0],
                'occupation': encoder['occupation'].transform([occupation])[0],
                'relationship': encoder['relationship'].transform([relationship])[0],
                'race': encoder['race'].transform([race])[0],
                'sex': encoder['sex'].transform([sex])[0],
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': encoder['native-country'].transform([native_country])[0],
            }

            input_df = pd.DataFrame([input_data])

            # Prediction
            prediction = model.predict(input_df)[0]

            # Decode label
            predicted_label = encoder['salary'].inverse_transform([prediction])[0]

            st.success(f"🎯 Predicted Sales Behavior: **{predicted_label}**")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
