import streamlit as st 
from PIL import Image
from classify import generate_desc,extract_features
from pickle import load
from keras.models import Model, load_model
import matplotlib.image as mpimg
import os


st.title("Upload + Classification Example")


tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34

filename = 'model_19.h5'
model = load_model(filename)

#####################################
test_imgs = os.listdir("test_images/")
test_img = st.selectbox(
    'Please Select a Test Image:',
    test_imgs
)

# Display and then predict on that image
fl_path = "test_images/"+test_img
# img = open_image(fl_path)

display_img = mpimg.imread(fl_path)
st.image(display_img, caption="Chosen Image", use_column_width=True)

st.write("")
with st.spinner("Identifying the Image"):
    time.sleep(5)
# label, prob = infer(img)
# st.success(f"Image : {label}, Confidence: {prob:.2f}%")


# uploaded_file = st.file_uploader("Choose an image...", type="jpg")


# path='C:\\Users\\acer\\Desktop\\Image Captioning\\Flickr8k_Dataset\\Flicker8k_Dataset\\'

up = extract_features(fl_path)

label = generate_desc(model, tokenizer, up, max_length)
# st.write('%s (%.2f%%)' % (label[1], label[2]*100))
st.write(label[8:-6])