# Importing the libries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# Loading the model
model = load_model('plant_disease.h5')

# Name of classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting title of app
st.title('Plant Disease Detection')
st.markdown('Upload an image of the plant leave')

# Upload the plant image
plant_image = st.file_uploader("Chosse an image... ", type='jpg')
submit = st.button('Predict')

# On predict button click
if submit:

    if plant_image is not None:
        # Convert the file into open cv
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1 )

        # Displaying the image
        st.image(opencv_image, channels='BGR')
        st.write(opencv_image.shape)

        # Resizing the  image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is " + result.split('-')[0]+ " leaf with " + result.split('-')[1]))