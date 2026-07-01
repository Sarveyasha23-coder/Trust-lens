import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    USE_XGB = False


# -------------------------
# PAGE
# -------------------------

st.set_page_config(
    page_title="Heart Failure Prediction  ",
   
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Failure Prediction")
st.caption("Developed by Sarveyasha")
st.write(
    "Predict heart disease risk using machine learning."
)


# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    # change path if needed
    df = pd.read_csv("heart.csv")

    return df


# -------------------------
# TRAIN MODEL
# -------------------------

@st.cache_resource
def train_model():

    df = load_data()

    target = "HeartDisease"

    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocess = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                num_cols
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                cat_cols
            )
        ]
    )

    if USE_XGB:

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            eval_metric="logloss"
        )

    else:

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, prob)

    return pipe, score


try:

    model, auc = train_model()

    st.success(
        f"Model ready • ROC-AUC: {auc:.3f}"
    )

except Exception as e:

    st.error(
        "Dataset not found."
    )

    st.code(
"""
Put this file beside app.py:

heart.csv
app.py
"""
    )

    st.stop()


# -------------------------
# INPUTS
# -------------------------

left, right = st.columns(2)

with left:

    age = st.slider(
        "Age",
        18,
        100,
        45
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest = st.selectbox(
        "Chest Pain Type",
        [
            "ATA",
            "NAP",
            "ASY",
            "TA"
        ]
    )

    bp = st.number_input(
        "Resting BP",
        80,
        250,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

    sugar = st.selectbox(
        "Fasting BS",
        [0, 1]
    )

with right:

    ecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST",
            "LVH"
        ]
    )

    hr = st.slider(
        "Max HR",
        60,
        220,
        150
    )

    angina = st.selectbox(
        "Exercise Angina",
        [
            "Y",
            "N"
        ]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.5,
        1.0
    )

    slope = st.selectbox(
        "ST Slope",
        [
            "Up",
            "Flat",
            "Down"
        ]
    )


# -------------------------
# PREDICT
# -------------------------

if st.button("Predict"):

    sample = pd.DataFrame([
        {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest,
            "RestingBP": bp,
            "Cholesterol": chol,
            "FastingBS": sugar,
            "RestingECG": ecg,
            "MaxHR": hr,
            "ExerciseAngina": angina,
            "Oldpeak": oldpeak,
            "ST_Slope": slope
        }
    ])

    pred = model.predict(sample)[0]

    prob = float(
        model.predict_proba(sample)[0][1]
    )

    st.divider()

    if pred == 1:

        st.error(
            f"High Risk ({prob:.1%})"
        )

    else:

        st.success(
            f"Low Risk ({prob:.1%})"
        )

    st.progress(prob)

    st.dataframe(
        sample,
        use_container_width=True
    )
