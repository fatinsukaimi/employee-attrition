import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier

# Load models and preprocessor
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.pkl")
    nn_model = tf.keras.models.load_model("nn_model.keras")
    xgb_model = joblib.load("hybrid_model.pkl")
    return preprocessor, nn_model, xgb_model

# Prediction function
def predict_attrition(data, preprocessor, nn_model, xgb_model):
    processed_data = preprocessor.transform(data)
    nn_predictions = nn_model.predict(processed_data)
    xgb_predictions = xgb_model.predict_proba(processed_data)[:, 1]
    hybrid_predictions = (nn_predictions.flatten() + xgb_predictions) / 2
    return (hybrid_predictions >= 0.5).astype(int), hybrid_predictions

# Main Streamlit app
def main():
    st.title("Employee Attrition Prediction App")
    st.write(
        """
        This app uses a hybrid Neural Network and XGBoost model to predict employee attrition.
        Upload your dataset to get predictions.
        """
    )

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(data.head())

        # Load models
        preprocessor, nn_model, xgb_model = load_models()

        # Predict attrition
        if st.button("Predict"):
            predictions, scores = predict_attrition(data, preprocessor, nn_model, xgb_model)
            data["Attrition Prediction"] = predictions
            data["Attrition Probability"] = scores
            st.write("Predictions:")
            st.write(data[["Attrition Prediction", "Attrition Probability"]])
            st.download_button(
                label="Download Predictions",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="attrition_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
