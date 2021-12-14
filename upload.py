import streamlit as st 
from PIL import Image
from classify import generate_desc,extract_features
from pickle import load
from keras.models import Model, load_model


st.title("Upload + Classification Example")


tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34

filename = 'model_19.h5'
model = load_model(filename)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    path='C:\\Users\\acer\\Desktop\\Image Captioning\\Flickr8k_Dataset\\Flicker8k_Dataset\\'
    st.write(uploaded_file.name)
    up = extract_features(path+uploaded_file.name)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = generate_desc(model, tokenizer, up, max_length)
    # st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    st.write(label[8:-6])