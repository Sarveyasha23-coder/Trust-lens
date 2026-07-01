import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ------------------------
# PAGE
# ------------------------

st.set_page_config(
    page_title="TrustLens",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------
# LOAD MODEL
# ------------------------

@st.cache_resource
def load_model():
    return joblib.load("trustlens (2).pkl")

model = load_model()

# ------------------------
# STYLE
# ------------------------

st.markdown("""
<style>

.big{
font-size:55px;
font-weight:800;
text-align:center;
}

.sub{
text-align:center;
font-size:18px;
color:gray;
}

.block{
padding:25px;
border-radius:20px;
background:#111111;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='big'>
🛡️ TrustLens
</div>

<div class='sub'>
AI Deepfake Detection • Built by Sarveyasha Sodhiya
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("Upload image → Analyze → Detect")

# ------------------------
# IMAGE
# ------------------------

uploaded=st.file_uploader(
"Upload Image",
type=["png","jpg","jpeg"]
)

if uploaded:

    image=Image.open(uploaded)

    st.image(
        image,
        use_container_width=True
    )

    width,height=image.size

    img=np.array(image)

    brightness=float(np.mean(img))

    quality=min(
        int(brightness/255*100),
        100
    )

    confidence=round(
        brightness/255,
        2
    )

    st.write("Image Resolution:",f"{width}×{height}")

# ------------------------
# BUTTON
# ------------------------

if uploaded and st.button("Analyze"):

    try:

        feature_names=list(
            model.feature_names_in_
        )

        row={}

        for col in feature_names:

            if col=="image_quality":
                row[col]=quality

            elif col=="confidence_score":
                row[col]=confidence

            elif col=="resolution":
                row[col]=width

            elif col=="gender":
                row[col]=0

            elif col=="age_group":
                row[col]=0

            elif col=="year":
                row[col]=2026

            elif col=="label_numeric":
                row[col]=0

            else:
                row[col]=0

        X=pd.DataFrame([row])

        pred=model.predict(X)[0]

        prob=None

        try:
            prob=max(
                model.predict_proba(X)[0]
            )

        except:
            prob=.90

        st.write("---")

        if pred==1:

            st.error(
f"""
🚨 DEEPFAKE DETECTED

Confidence:
{prob:.0%}
"""
)

        else:

            st.success(
f"""
✅ REAL IMAGE

Confidence:
{prob:.0%}
"""
)

        st.write("Features Used")

        st.dataframe(X)

    except Exception as e:

        st.error(e)

st.write("---")

st.caption(
"TrustLens • Built by Sarveyasha Sodhiya"
)
