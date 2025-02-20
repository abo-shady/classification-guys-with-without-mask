import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
import streamlit as st
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the model
model = models.load_model('model.h5')

# Streamlit app title
st.title("Image Classification App")

# File uploader
uploaded_file = st.file_uploader(label="Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image using PIL
    image = Image.open(uploaded_file)
    
    # Convert the image to a numpy array
    img = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)    
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR (OpenCV format)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by your model
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Button to trigger prediction
    if st.button('Predict'):
        # Make a prediction
        with st.spinner('Predicting...'):

            pred = model.predict(np.array([img]))  # Add batch dimension and predict
            index = np.argmax(pred)  # Get the index of the highest probability class

            # Map the index to a class label
            class_names = ['with_mask', 'without_mask']  # Replace with your actual class names
            st.write('Predicted class:', class_names[index])
else:
    st.write("Please upload an image to proceed.")