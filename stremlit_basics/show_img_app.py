import streamlit as st
from PIL import Image

# Create an upload button and save the uploaded file
uploaded_file = st.file_uploader("Choose a TIFF image...", type=['tif', 'tiff'])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded TIFF Image')
else:
    # Display a message when no image is uploaded
    st.write("Please upload a TIFF image.")

