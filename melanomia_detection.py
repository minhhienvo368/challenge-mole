import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import keras
# Title of the page
st.set_page_config(
        page_title="Melanomia App",
)

def cancer_classification(img, file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = keras.models.load_model(file, compile=False)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction)

# Title and introduction
st.title(" Melanomia Detection App")

# file upload and handling logic
uploaded_file = st.file_uploader("Upload a close-up image (.JPEG) of skin lesion", 
                                    type=["jpeg","jpg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    st.write("")
    st.subheader("Result")

    # Load models
    label_mole = cancer_classification(image, 'keras_Mobile_model')[0]

    # if image is not a X-Ray image
    # x ray image
    # normal lungs
    if label_mole == 0:
        st.write("This X Ray looks normal. This X Ray depicts __CLEAR LUNGS__.")
    # infected lungs
    else:
        st.write()


# Cleans Streamlit layout
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
