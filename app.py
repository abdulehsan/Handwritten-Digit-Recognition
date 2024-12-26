import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


file_path = Path(__file__).parent / 'digit_recognition_model.h5'

model = tf.keras.models.load_model(file_path)

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit for recognition!")

uploaded_file = st.file_uploader("Upload an image for detailed analysis", type=["png", "jpg", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert('L')
    image = np.array(image)
    image = cv2.resize(image, (28, 28)) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    probabilities = prediction.flatten()

    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(range(10), probabilities, color='blue')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    predicted_digit = np.argmax(probabilities)
    st.write(f"Predicted Digit: {predicted_digit}")
