import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os 
import pandas as pd
from tensorflow.keras.applications.vgg16 import preprocess_input

os.chdir(r'C:\Test')
# Load the model
model = tf.keras.models.load_model('vgg16_BT_detect_ver11.h5')

# Define the labels (adjust these based on your model's output classes)
labels = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No tumor",
    3: "Pituitary tumor"
}

# Function to preprocess the image using the custom preprocessing method
def preprocess_image(image, target_size):
    # Convert PIL image to RGB (ensures 3 channels)
    image = image.convert("RGB")
    # Convert PIL image to numpy array
    image = np.array(image)

    # Resize the image to the target size
    image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    # Apply VGG-16 preprocessing
    image = preprocess_input(image)

    # Expand dimensions to match the model's input shape (1, height, width, 3)
    image = np.expand_dims(image, axis=0)

    return image

# Streamlit app layout
st.title("Check your MRI image for Tumor Detection and Classification.")
st.write("Here upload an MRI image from your device.")

# Image upload
uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Resize and display the image
    resized_image = image.resize((100, 100)) 
    st.image(resized_image, caption='Uploaded MRI Image')

    # Preprocess the image using the custom function
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    # Display image details
    st.write(f"**Image Shape:** {image.size}")

    # Display the result in a table format
    st.write("### Prediction Results")
    prediction_table = []
    for i, label in labels.items():
        prediction_table.append({
            "Classes": label,
            "Prediction": f"{prediction[0][i] * 100:.2f}%"
        })

    # Highlight the predicted class
    df = pd.DataFrame(prediction_table)
    st.table(df.style.apply(lambda x: ['background-color: lightblue' if x.name == predicted_class else '' for _ in x], axis=1))
