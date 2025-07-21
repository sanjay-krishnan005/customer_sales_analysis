# 📊 Customer Sales Prediction

This project predicts customer sales behavior (e.g., income category) using machine learning based on demographic features like age, workclass, education, and occupation. It includes a deployed Streamlit web app for real-time predictions.

## 🚀 Features

- Real-time prediction using a trained ML model
- Label encoding for categorical variables
- Model and encoder saved using Joblib
- Interactive UI built with Streamlit

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Joblib

## 🧪 Input Features

- Age
- Workclass
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain / Loss
- Hours per Week
- Native Country

## 🧠 Model

- Classification Model (KNN - K Nearest Neighbor)

## 📂 Project Structure

├── customer_sales_analysis.ipynb # Notebook with data analysis & training
├── app.py # Streamlit web app
├── model.pkl # Trained model
├── label_encoders.pkl # Saved encoders
├── requirements.txt # Dependencies
└── README.md # Project documentation

## 📦 Installation

pip install -r requirements.txt
streamlit run app.py

## Future Improvements

- Add data visualization in Streamlit
- Improve model accuracy with hyperparameter tuning
- Add SHAP or LIME for explainability


