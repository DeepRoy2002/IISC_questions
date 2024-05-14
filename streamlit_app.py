import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import keras

# Load your saved model
model = tf.keras.models.load_model('C:/Users/anirb/IISC questions/cifar10/model_1.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of your model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Define custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS
local_css("style.css")

def main():
    st.title("🖼️ Image Classification App")
    st.markdown("---")

    st.write("Upload an image for classification.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        prediction = predict(image)
        st.write("---")
        st.subheader("Prediction")
        st.write(prediction)

if __name__ == "__main__":
    main()

