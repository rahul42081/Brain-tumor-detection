# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import model_from_json
model=model_from_json(open('tumor.json','r').read())
model.load_weights('Tumor.h5')
def detect(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grey=cv2.cvtColor(opencvImage,cv2.COLOR_BGR2GRAY)
    grey=cv2.resize(grey,(100,100))
    return grey,opencvImage
def about():
	st.write(
		'''
		**Tensorflow and Opencv** are libraries
		used for image processing  

Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')

m={0:'Non-Tumored',1:'Tumored'}
def main(model):
    st.title("Brain Tumor Detection App :sunglasses: ")
    st.write("**Using the Tensorflow and Opencv**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
    			result,opencvImage = detect(image=image)
    			predictions=model.predict(result.reshape(1,100,100,1))
    			st.image(opencvImage,caption=m[np.argmax(predictions[0])]+' Image')
    			st.success("{}\n".format(m[np.argmax(predictions[0])]))

    elif choice == "About":
    	about()
    	
if __name__ == "__main__":
    main(model)
