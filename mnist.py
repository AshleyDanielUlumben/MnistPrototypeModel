import pandas as pd
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.cache(allow_output_mutation=True)
def model():
    return keras.models.load_model('mnist.h5')

model = model()

page_bg_img = '''
<style>
body {
background-image: url(https://i.pinimg.com/originals/04/30/2c/04302cf312b3484b3db51228f2eb4910.jpg);
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.header("Final Project")
st.write("The Prediction Model based on MNIST Dataset")
st.write('''
    Draw a digit from 0-9 in the box below.
    Click Predict button to see the prediction.
''')
st.write('''_(It is not very accurate.)_''')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader('Canvas')
    # Specify canvas parameters in application
    st.sidebar.subheader("Stroke Width:")
    stroke_width = st.sidebar.slider("Works best with 20", 1, 30, 20)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=200,
        height=200,
        drawing_mode="freedraw",
        key='canvas'
    )

with col2:
    if canvas_result.image_data is not None:
        image = cv.resize(canvas_result.image_data.astype('uint8'), (28,28))
        rescaled_image = cv.resize(image, (200,200), interpolation=cv.INTER_NEAREST)
        st.subheader("Model Input")
        st.image(rescaled_image)


if st.button('Predict'):
    input_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    pred_proba = model.predict(input_image.reshape(1, 28, 28))
    df = pd.DataFrame(data = pred_proba.reshape(-1,1), columns=['Probabilities'])
    st.subheader("Output Probabilities")
    st.bar_chart(df)