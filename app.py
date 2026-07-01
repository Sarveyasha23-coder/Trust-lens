import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
MODEL_PATH = "heart_failure_best_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------
# HEADER
# -------------------------
st.title("❤️ Heart Failure Prediction")
st.write(
    "Predict the probability of heart disease using patient health information."
)

# -------------------------
# INPUT FORM
# -------------------------

col1, col2 = st.columns(2)

with col1:

    Age = st.slider(
        "Age",
        18,
        100,
        45
    )

    Sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    RestingBP = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

    FastingBS = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

with col2:

    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    MaxHR = st.slider(
        "Max Heart Rate",
        60,
        220,
        150
    )

    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        ["Y", "N"]
    )

    Oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.5,
        1.0
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# -------------------------
# CREATE INPUT
# -------------------------

input_df = pd.DataFrame([{
    "Age": Age,
    "Sex": Sex,
    "ChestPainType": ChestPainType,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "Oldpeak": Oldpeak,
    "ST_Slope": ST_Slope
}])


# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict"):

    try:

        pred = model.predict(input_df)[0]

        prob = float(
            model.predict_proba(input_df)[0][1]
        )

        st.divider()

        if pred == 1:

            st.error(
                f"High Risk Detected\n\nProbability: {prob:.2%}"
            )

        else:

            st.success(
                f"Low Risk\n\nProbability: {prob:.2%}"
            )

        st.subheader("Risk Meter")

        st.progress(prob)

        st.metric(
            "Heart Disease Probability",
            f"{prob:.1%}"
        )

        st.subheader("Entered Values")

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as e:

        st.exception(e)


# -------------------------
# FOOTER
# -------------------------

st.markdown("---")

st.caption(
    "Model: Trained using Heart Failure Prediction Dataset"
)
