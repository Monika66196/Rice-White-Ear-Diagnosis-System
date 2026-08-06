import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from labels import CLASS_NAMES
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Rice White Ear Diagnosis",
    page_icon="🌾",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "Rice_Final_Model.keras",
        compile=False
    )
    return model

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Rice White Ear Diagnosis")

option = st.sidebar.radio(
    "Choose Image Source",
    [
        "Upload Image",
        "Use Sample Images"
    ]
)

# -----------------------------
# Title
# -----------------------------
st.title("🌾 Rice White Ear Diagnosis System")

st.write(
"""
Upload a rice white ear image or select a sample image.

The model predicts:

• Healthy

• Severity 1–25%

• Severity 26–50%

• Severity 51–75%

• Severity 76–100%
"""
)

image = None

# -----------------------------
# Upload Option
# -----------------------------
if option == "Upload Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

# -----------------------------
# Sample Images
# -----------------------------
else:

    sample_folder = "samples"

    sample_images = [
        f for f in os.listdir(sample_folder)
        if f.endswith((".jpg",".png",".jpeg"))
    ]

    selected = st.selectbox(
        "Choose Sample",
        sample_images
    )

    image = Image.open(
        os.path.join(sample_folder, selected)
    ).convert("RGB")

# -----------------------------
# Prediction
# -----------------------------
if image is not None:

    st.image(image, use_container_width=True)

    img = image.resize((224,224))

    img = np.array(img)/255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)*100

    st.success(
        f"Prediction : {CLASS_NAMES[predicted_class]}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )

    st.subheader("Class Probabilities")

    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(prediction[0][i]))
        st.write(f"{cls}: {prediction[0][i]*100:.2f}%")

        yield_loss_data = {
    "Healthy": {
        "Relative Grain Yield (%)": 100.00,
        "Yield Loss (%)": 0.00
    },
    "Severity_1_25": {
        "Relative Grain Yield (%)": 85.46,
        "Yield Loss (%)": 14.54
    },
    "Severity_26_50": {
        "Relative Grain Yield (%)": 67.41,
        "Yield Loss (%)": 32.59
    },
    "Severity_51_75": {
        "Relative Grain Yield (%)": 43.86,
        "Yield Loss (%)": 56.14
    },
    "Severity_76_100": {
        "Relative Grain Yield (%)": 20.00,
        "Yield Loss (%)": 80.00
    }
}

        
