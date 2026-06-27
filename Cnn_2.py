import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Fruit Freshness AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
}

.hero {
    text-align: center;
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white;
}

.info-box {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.fresh {
    background: #dcfce7;
    color: #15803d;
}

.rotten {
    background: #fee2e2;
    color: #dc2626;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_fresh_rotten_classifier.keras"
    )

model = load_model()

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🍎 Fruit Freshness AI</h1>
    <p>Deep Learning Powered Fruit Quality Detection System</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# INFO SECTION
# ---------------------------------------------------
st.markdown("""
<div class="info-box">
    <h2>⚡ AI-Powered Fruit Quality Detection</h2>
    <p style="font-size:18px;">
        Upload a fruit image and instantly classify it as
        <b>Fresh</b> or <b>Rotten</b> using a Deep Learning CNN model.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# MAIN SECTION
# ---------------------------------------------------
left, right = st.columns([1, 1])

with left:

    uploaded = st.file_uploader(
        "📤 Upload Fruit Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded Fruit",
            use_container_width=True
        )

with right:

    if uploaded:

        st.subheader("Prediction")

        if st.button("🔍 Analyze the Fruit"):

            with st.spinner("Running AI Analysis..."):

                img = preprocess(image)

                prediction = model.predict(
                    img,
                    verbose=0
                )[0][0]

                time.sleep(0.5)

                if prediction >= 0.5:

                    confidence = prediction * 100

                    st.markdown(f"""
                    <div class='result-box rotten'>
                    ❌ Rotten Fruit
                    <br><br>
                    Confidence: {confidence:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    confidence = (1 - prediction) * 100

                    st.markdown(f"""
                    <div class='result-box fresh'>
                    ✅ Fresh Fruit
                    <br><br>
                    Confidence: {confidence:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(float(max(prediction, 1 - prediction)))

                st.success("Prediction Complete")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.caption(
    "Built with TensorFlow • Streamlit • CNN Deep Learning"
)