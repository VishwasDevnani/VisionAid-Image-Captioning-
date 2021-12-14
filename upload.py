import streamlit as st 
from PIL import Image
from classify import generate_desc,extract_features
from pickle import load
from keras.models import Model, load_model
import matplotlib.image as mpimg
import os
import time


option = st.radio('', ['Choose a Sample', 'Upload your own image'])

st.title("Image Captioning")
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34

filename = 'model_19.h5'
model = load_model(filename)
if option == 'Choose a Sample':
    

    #####################################
    test_imgs = os.listdir("test_images/")
    test_img = st.selectbox(
        'Please Select a Test Image:',
        test_imgs
    )


    fl_path = "test_images/"+test_img


    display_img = mpimg.imread(fl_path)
    st.image(display_img, caption="Chosen Image", use_column_width=True)

    st.write("")
    up = extract_features(fl_path)
    with st.spinner("Identifying whats going on in the image"):
        time.sleep(5)


    

    label = generate_desc(model, tokenizer, up, max_length)
    st.success(label[8:-6])

else:
    uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer()) 
        # img = image.pil2tensor(img, np.float32).div_(255)
        # img = image.Image(img)
        st.write("")
        up = extract_features("tempDir//"+uploaded_file.name)
        with st.spinner("Identifying whats going on in the image"):
            time.sleep(5)
        

        label = generate_desc(model, tokenizer, up, max_length)
        st.write(label[8:-6])
        
        # st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")