import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import json

IMAGE_SIZE = 128

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="PokéVision",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Cache ----------------
@st.cache_resource
def load_model():
    return keras.models.load_model("pokemon_classifier.keras")


@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)


model = load_model()
class_names = load_classes()

# ---------------- Sidebar ----------------
st.sidebar.title("📖 Model Information")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🧠 Model")
st.sidebar.write("**Custom Convolutional Neural Network (CNN)**")

st.sidebar.markdown("### 📊 Dataset")
st.sidebar.write("**Pokémon Generation I Dataset**")

st.sidebar.markdown("### 🔢 Classes")
st.sidebar.write(f"**{len(class_names)} Pokémon**")

st.sidebar.markdown("### 🖼️ Input Size")
st.sidebar.write(f"**{IMAGE_SIZE} × {IMAGE_SIZE} pixels**")

st.sidebar.markdown("### 🎯 Validation Accuracy")
st.sidebar.write("**~93%**")

st.sidebar.markdown("---")
st.sidebar.caption("Developed using TensorFlow, Keras and Streamlit")

# ---------------- Header ----------------
st.title("⚡ PokéVision")

st.subheader("AI-Powered Pokémon Image Classifier")

st.info("""
🎯 **Supports only Generation I Pokémon (Kanto Region)**

This application uses a **Convolutional Neural Network (CNN)** trained exclusively on the **original 151 Pokémon (Pokédex #001–#151)**.

⚠️ **Please Note:** Images of Pokémon from later generations, fan art with heavy modifications, or non-Pokémon objects are outside the model's training scope and may produce inaccurate predictions.
""")

st.write("📤 Upload a Pokémon image below to begin classification.")

st.markdown("---")

# ---------------- Upload ----------------
uploaded_file = st.file_uploader(
    "Choose a Pokémon Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(image, use_container_width=True)

    # ---------------- Prediction ----------------
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    probs = prediction[0]

    predicted_index = np.argmax(probs)
    confidence = probs[predicted_index] * 100

    with col2:

        st.subheader("🎯 Prediction")

        st.success(f"## {class_names[predicted_index]}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(probs[predicted_index]))

        st.markdown("---")

        st.subheader("🏆 Top 5 Predictions")

        top5 = np.argsort(probs)[::-1][:5]

        for idx in top5:

            st.write(
                f"**{class_names[idx]}** — {probs[idx] * 100:.2f}%"
            )

            st.progress(float(probs[idx]))

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "⚡ PokéVision • Deep Learning Image Classification Project • "
    "Built with TensorFlow, Keras & Streamlit"
)