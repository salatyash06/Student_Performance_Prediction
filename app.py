import streamlit as st
import pandas as pd
import joblib

st.title("Student Final Score Prediction")

model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")


#Step 30 
hours_studied = st.number_input(
    "Hours Studied per Week",
    min_value=0.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0
)

previous_scores = st.number_input(
    "Previous Examination Score",
    min_value=0.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0
)

tutoring_sessions = st.number_input(
    "Tutoring Sessions",
    min_value=0
)

#Step 31
input_data = pd.DataFrame({
    "Hours_Studied": [hours_studied],
    "Attendance": [attendance],
    "Previous_Scores": [previous_scores],
    "Sleep_Hours": [sleep_hours],
    "Tutoring_Sessions": [tutoring_sessions]
})

#Step 32
input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(
    columns=features,
    fill_value=0
)

#Step 33
if st.button("Predict Final Score"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Final Exam Score: {prediction[0]:.2f}"
    )