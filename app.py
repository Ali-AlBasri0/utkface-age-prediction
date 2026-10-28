import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model

st.title('Age Prediction')

model = load_model('utkface_age_cnn.keras')

photo = st.camera_input('Take a photo (face in the center)')

if photo is not None:
    img = Image.open(photo).convert('RGB')     # color
    img = ImageOps.fit(img, (100, 100))        # center crop + resize to 100x100

    age = model.predict(np.array(img).reshape(1, 100, 100, 3), verbose=0)[0][0]
    age = max(0, int(round(age)))              # normal whole number

    st.subheader('Predicted age: ' + str(age))
