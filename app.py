import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("Effiicientnetv2b2.keras")

# Class names (make sure this matches your training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title
st.title("🗑️ Garbage Classifier")

# Description
st.write("Upload an image to classify it into one of the following categories: **cardboard**, **glass**, **metal**, **paper**, **plastic**, or **trash**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"**{predicted_class}** with {confidence * 100:.2f}% confidence")
