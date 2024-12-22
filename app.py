import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# print(cv2.__version__)

file_path = Path(__file__).parent / 'digit_recognition_model.h5'
# Load the model
model = tf.keras.models.load_model(file_path)

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit for recognition!")

# File uploader
uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    st.write(f"Predicted Digit: {digit}")
