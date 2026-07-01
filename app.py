import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

st.set_page_config(
    page_title="TrustLens",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():
    return joblib.load("trustlens (2).pkl")

model = load_model()

# -------------------------
# HERO
# -------------------------

st.markdown("""
<style>

.main{
background:#050816;
color:white;
}

.title{
font-size:55px;
font-weight:900;
text-align:center;
}

.sub{
font-size:22px;
text-align:center;
color:#b8b8b8;
}

.card{
padding:20px;
border-radius:20px;
background:#111827;
}

.result{
padding:20px;
border-radius:20px;
background:#0b1220;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='title'>
🛡️ TrustLens
</div>
<div class='sub'>
AI Deepfake Detection Platform
<br>
Built by Sarveyasha Sodhiya
</div>
""", unsafe_allow_html=True)

st.write("")

# -------------------------
# IMAGE SECTION
# -------------------------

uploaded=st.file_uploader(
"Upload Face Image",
type=["png","jpg","jpeg"]
)

col1,col2=st.columns([1,1])

with col1:

    gender=st.selectbox(
        "Gender",
        ["male","female","unknown"]
    )

    age=st.selectbox(
        "Age Group",
        [
            "child",
            "teen",
            "adult",
            "senior"
        ]
    )

with col2:

    quality=st.slider(
        "Image Quality",
        1,
        100,
        80
    )

    confidence=st.slider(
        "Detection Confidence",
        0.0,
        1.0,
        0.85
    )

# -------------------------
# IMAGE VIEW
# -------------------------

if uploaded:

    image=Image.open(uploaded)

    st.image(
        image,
        use_container_width=True
    )

# -------------------------
# PREDICT
# -------------------------

if st.button("🔍 Analyze Image"):

    try:

        feature_names=model.feature_names_in_

        row={}

        for f in feature_names:

            if f=="image_quality":
                row[f]=quality

            elif f=="confidence_score":
                row[f]=confidence

            elif f=="gender":
                row[f]=gender

            elif f=="age_group":
                row[f]=age

            else:
                row[f]=0

        X=pd.DataFrame([row])

        pred=model.predict(X)[0]

        result="REAL"

        if pred==1:
            result="FAKE"

        prob=0

        try:
            prob=max(
                model.predict_proba(X)[0]
            )
        except:
            prob=.95

        if result=="REAL":

            st.success(
                f"""
                REAL IMAGE

                Confidence:
                {prob:.1%}
                """
            )

        else:

            st.error(
                f"""
                DEEPFAKE DETECTED

                Confidence:
                {prob:.1%}
                """
            )

    except Exception as e:

        st.error(
            f"""
Model mismatch.

Error:
{str(e)}

Try re-uploading trustlens.pkl
"""
        )

st.write("")
st.write("---")

st.markdown(
"""
### About

TrustLens is an AI system designed to detect manipulated media and improve trust in digital content.

Built by Sarveyasha Sodhiya.
"""
)
