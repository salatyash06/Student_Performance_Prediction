
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Light theme and custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #FFFFFF;
        color: #1F2937;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
    }

    /* Content width and spacing */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headings and labels */
    h1, h2, h3, p, label {
        color: #1F2937;
    }

    /* Slider spacing */
    [data-testid="stSlider"] {
        padding-bottom: 1.3rem;
    }

    /* Purple slider controls */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #5145E5;
        border-color: #5145E5;
    }

    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #5145E5;
    }

    /* Predict Score button */
    div.stButton > button {
        width: 100%;
        min-height: 58px;
        padding: 0.8rem 1rem;

        background-color: #5145E5 !important;
        color: #FFFFFF !important;

        border: 1px solid #5145E5 !important;
        border-radius: 12px;

        font-size: 1.15rem;
        font-weight: 500;

        transition: background-color 0.2s ease;
    }

    /* Force white text on all button contents */
    div.stButton > button *,
    div.stButton > button p,
    div.stButton > button span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Hover state */
    div.stButton > button:hover {
        background-color: #4338CA !important;
        color: #FFFFFF !important;
        border-color: #4338CA !important;
    }

    div.stButton > button:hover *,
    div.stButton > button:hover p,
    div.stButton > button:hover span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Active and focused states */
    div.stButton > button:active,
    div.stButton > button:focus,
    div.stButton > button:focus-visible {
        color: #FFFFFF !important;
        border-color: #5145E5 !important;
    }

    div.stButton > button:active *,
    div.stButton > button:focus *,
    div.stButton > button:focus-visible * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Result message styling */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Divider */
    hr {
        border-color: #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("Student Performance Prediction")
st.caption(
    "Adjust the values below to predict your final exam score."
)

st.markdown("---")

# --------------------------------------------------
# Load trained model and feature names
# --------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features


try:
    model, features = load_model_files()
except FileNotFoundError:
    st.error(
        "Model files were not found. Please check that "
        "'models/model.pkl' and 'models/features.pkl' exist."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# Student input fields
# --------------------------------------------------
hours_studied = st.slider(
    "Study hours / week",
    min_value=0,
    max_value=100,
    value=20,
    step=1,
    help="Number of hours spent studying per week.",
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=1,
    help="Percentage of classes attended.",
)

previous_scores = st.slider(
    "Previous exam score",
    min_value=0,
    max_value=100,
    value=75,
    step=1,
    help="Score obtained in a previous examination.",
)

sleep_hours = st.slider(
    "Sleep hours / night",
    min_value=0,
    max_value=24,
    value=7,
    step=1,
    help="Average number of hours slept per night.",
)

tutoring_sessions = st.slider(
    "Tutoring sessions",
    min_value=0,
    max_value=20,
    value=1,
    step=1,
    help="Number of tutoring sessions attended.",
)

st.markdown("---")

# --------------------------------------------------
# Prediction button and result
# --------------------------------------------------
if st.button(
    "Predict Score",
    type="primary",
    use_container_width=True,
):
    try:
        input_data = pd.DataFrame(
            {
                "Hours_Studied": [hours_studied],
                "Attendance": [attendance],
                "Previous_Scores": [previous_scores],
                "Sleep_Hours": [sleep_hours],
                "Tutoring_Sessions": [tutoring_sessions],
            }
        )

        # Align inputs with the trained model's feature columns
        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=features,
            fill_value=0,
        )

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Final Exam Score: {prediction[0]:.2f}"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
