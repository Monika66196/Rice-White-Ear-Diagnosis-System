import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from labels import CLASS_NAMES
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Rice White Ear Diagnosis",
    page_icon="🌾",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "Rice_Final_Model.keras",
        compile=False
    )

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🌾 Rice White Ear Diagnosis")

option = st.sidebar.radio(
    "Select Image Source",
    [
        "Upload Image",
        "Use Sample Image"
    ]
)

# --------------------------------------------------
# Main Title
# --------------------------------------------------
st.title("🌾 Rice White Ear Diagnosis System")

st.markdown("""
This application predicts the severity of **Rice White Ear Disease** using a trained CNN model.

### Supported Classes
- Healthy
- Severity 1–25%
- Severity 26–50%
- Severity 51–75%
- Severity 76–100%
""")

image = None

# --------------------------------------------------
# Upload Image
# --------------------------------------------------
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# --------------------------------------------------
# Sample Images
# --------------------------------------------------
else:

    sample_folder = "samples"

    if os.path.exists(sample_folder):

        sample_images = sorted([
            f for f in os.listdir(sample_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(sample_images) > 0:

            selected = st.selectbox(
                "Choose Sample Image",
                sample_images
            )

            image = Image.open(
                os.path.join(sample_folder, selected)
            ).convert("RGB")

        else:
            st.warning("No sample images found.")

    else:
        st.warning("Samples folder not found.")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    predicted_label = CLASS_NAMES[predicted_class]

    confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction : {predicted_label}")

    st.info(f"Confidence : {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for i, cls in enumerate(CLASS_NAMES):

        st.write(f"**{cls}**")

        st.progress(float(prediction[0][i]))

        st.write(f"{prediction[0][i]*100:.2f}%")

            # ==================================================
    # Yield Loss Estimation
    # ==================================================

    yield_loss_data = {
        "Healthy": {
            "Relative Grain Yield": 100.00,
            "Yield Loss": 0.00
        },
        "Severity 1-25%": {
            "Relative Grain Yield": 85.46,
            "Yield Loss": 14.54
        },
        "Severity 26-50%": {
            "Relative Grain Yield": 67.41,
            "Yield Loss": 32.59
        },
        "Severity 51-75%": {
            "Relative Grain Yield": 43.86,
            "Yield Loss": 56.14
        },
        "Severity 76-100%": {
            "Relative Grain Yield": 20.00,
            "Yield Loss": 80.00
        }
    }

    st.markdown("---")
    st.subheader("🌾 Yield Loss Estimation")

    result = yield_loss_data.get(predicted_label)

    if result is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Relative Grain Yield",
                f"{result['Relative Grain Yield']:.2f}%"
            )

        with col2:
            st.metric(
                "Estimated Yield Loss",
                f"{result['Yield Loss']:.2f}%"
            )

        st.progress(result["Yield Loss"] / 100)

        st.caption(
            f"Estimated Yield Loss: {result['Yield Loss']:.2f}%"
        )

    else:
        st.warning("Yield loss information not available.")

    st.markdown("---")


  
        
