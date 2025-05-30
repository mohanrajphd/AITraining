import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Image Classification App", page_icon="📸")

# Title and description
st.title("Image Classification with VGG16")
st.write("Upload an image, and the model will predict its class using VGG16 pre-trained on ImageNet.")

# Load the pre-trained VGG16 model
@st.cache_resource
def load_model():
    return VGG16(weights='imagenet')

model = load_model()

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for VGG16
    image = image.resize((224, 224))  # VGG16 input size
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Convert RGBA to RGB if needed
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(image_array)
        decoded_preds = decode_predictions(predictions, top=3)[0]
    
    # Display predictions
    st.subheader("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"{i+1}. {label}: {score*100:.2f}%")