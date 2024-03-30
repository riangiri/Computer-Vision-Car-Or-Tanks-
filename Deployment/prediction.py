import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Function memproses gambar dengan limit pixel 255x255 
def process_image(image, target_size=(255, 255)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Function untuk membuat prediksi binary clasification
def predict(image, model):
    preprocess = process_image(image)
    prediction = model.predict(preprocess)
    if prediction[0, 0] > 0.5:
        return "Tank"
    else:
        return "Car"

# deploy streamlit Web app
def main():
    st.title("Car or Tank Image Classifier")
    st.sidebar.title("Upload Image/Gambar")

    uploaded_file = st.sidebar.file_uploader("Pilih Gambar (Choose an image)", type=["jpg", "jpeg"])

    # sample photo/image
    sample_photos = ["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg"]

    if uploaded_file is not None:
        # Display gambar yang telah di upload
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load model terbaik
        model_location = "Car-tanks-model.h5"
        model = load_model(model_location)

        # Convert image ke arry
        image_array = np.array(image)

        # Buat prediksi
        with st.spinner('Predicting...'):
            predict_class = predict(image_array, model)

        # Show hasil prediksi
        st.write(f"# **Hasil Predict Klasifikasi : {predict_class}**")

    # Display sample photos
    st.sidebar.title("Sample Photos (Click and Drag photos/image to browse file)")
    for sample_photo in sample_photos:
        sample_image = Image.open(sample_photo)
        st.sidebar.image(sample_image, caption=sample_photo, width=300)

if __name__ == "__main__":
    main()