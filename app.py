import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="TrustLens",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "trustlens (2).pkl"

model = joblib.load(MODEL_PATH)

# -------------------
# CUSTOM CSS
# -------------------

st.markdown("""
<style>

.main{
background:linear-gradient(135deg,#08111f,#132238);
color:white;
}

.title{
font-size:55px;
font-weight:800;
text-align:center;
}

.subtitle{
text-align:center;
font-size:20px;
color:#cccccc;
}

.metric{
padding:20px;
border-radius:20px;
background:#18283d;
}

.stButton>button{
width:100%;
height:60px;
font-size:20px;
border-radius:20px;
background:#2e6bff;
color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# HEADER
# -------------------

st.markdown(
"""
<div class='title'>
🛡️ TrustLens
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Built by Sarveyasha Sodhiya
<br>
Industry-Ready AI Authenticity Engine
</div>
""",
unsafe_allow_html=True
)

st.divider()

# -------------------
# INPUTS
# -------------------

st.header("Deepfake Risk Analyzer")

col1,col2=st.columns(2)

with col1:

    confidence=st.slider(
        "Confidence Score",
        0.0,
        1.0,
        0.7
    )

    year=st.slider(
        "Year",
        2020,
        2030,
        2026
    )

with col2:

    quality=st.selectbox(
        "Image Quality",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    difficulty=st.selectbox(
        "Detection Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )

# encoding

quality_map={
"Low":0,
"Medium":1,
"High":2
}

difficulty_map={
"Easy":0,
"Medium":1,
"Hard":2
}

# -------------------
# PREDICT
# -------------------

if st.button("Analyze Trust Score"):

    X=pd.DataFrame([[
        confidence,
        quality_map[quality],
        difficulty_map[difficulty],
        year
    ]])

    with st.spinner("Scanning authenticity..."):
        time.sleep(2)

        pred=model.predict(X)[0]

    st.divider()

    if pred==1:

        st.error(
            """
            ⚠️ Potential Deepfake Detected
            """
        )

        st.progress(85)

    else:

        st.success(
            """
            ✅ Appears Authentic
            """
        )

        st.progress(20)

# -------------------
# FOOTER
# -------------------

st.divider()

st.markdown("""
### Why TrustLens?

✔ Enterprise-style UI

✔ Instant AI inference

✔ Lightweight deployment

✔ Global product design

✔ Ready for portfolio & startup demo

""")
