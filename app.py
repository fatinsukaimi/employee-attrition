import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and preprocessor
nn_model = load_model("nn_model.keras")
hybrid_model = joblib.load("hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title
st.title("Employee Attrition Prediction")

# Sidebar Inputs
st.sidebar.header("Employee Features")
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.number_input("Monthly Income (e.g., 5000)", min_value=1000, step=500)
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 0, 50, 10)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)

# Convert input into DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "PercentSalaryHike": [percent_salary_hike],
    "YearsWithCurrManager": [years_with_curr_manager],
    "JobInvolvement": [job_involvement],
})

# Preprocess input data
input_array = preprocessor.transform(input_data)

# Predict using Neural Network
nn_predictions = nn_model.predict(input_array).flatten()

# Create hybrid features
input_hybrid = np.column_stack((input_array, nn_predictions))

# Predict using Hybrid NN-XGBoost
hybrid_predictions = hybrid_model.predict(input_hybrid)

# Display predictions
st.subheader("Prediction Results")
if hybrid_predictions[0] == 1:
    st.write("The employee is likely to leave the company.")
else:
    st.write("The employee is likely to stay in the company.")
