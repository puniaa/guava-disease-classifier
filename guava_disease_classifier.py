import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model (ensure it's saved in the same directory or provide the correct path)
model = load_model('guava_disease_model.h5')

# Streamlit app interface
st.title("Guava Disease Classifier")
st.write("Upload an image of a guava leaf to predict its disease.")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image...", type="png")

# When an image is uploaded
if uploaded_image is not None:
    # Open the image
    img = image.load_img(uploaded_image, target_size=(512, 512))  # Resize the image to match input size
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand the array to add batch size dimension
    img_array /= 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Define class names based on your model’s output
    class_indices = {0: 'Healthy', 1: 'Anthracnose', 2: 'Fruit Flies'}
    predicted_label = class_indices[predicted_class]

    # Show the image and prediction
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {predicted_label}")

