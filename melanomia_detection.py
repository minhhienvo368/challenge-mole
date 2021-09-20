import streamlit as st
from PIL import Image
# Title of the page
st.set_page_config(
        page_title="Melanomia App",
)

# Title and introduction
st.title(" Melanomia Detection App")

# file upload and handling logic
uploaded_file = st.file_uploader("Upload a close-up image (.JPEG) of skin lesion", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    st.write("")
    st.subheader("Result")

    # Load models
    label_mole = x_ray_classification(image, 'keras_Mobile_model')
    label_pneumonia = pneumonia_classification(image, 'keras_model_pneumonia.h5')


    # if image is not a X-Ray image
    if label_x_ray == 0:
        st.write("""
                This doesn't look like a X Ray image of lungs.
                \nPlease restart with a proper image.
                """)
    # x ray image
    else:
        # normal lungs
        if label_pneumonia == 0:
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
