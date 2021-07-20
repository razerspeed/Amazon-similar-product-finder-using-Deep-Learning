import streamlit as st
from saving_test_csv import create_dataset
from model_save import predict
import pandas as pd
from PIL import Image
from download import download_from_drive



app_formal_name = "Image Caption"
# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_formal_name,
)


st.title("Amazon Similar Product Finder using Deep Learning")

st.markdown('#### Do you scan online retailers in search of the best deals? This webapp can help you .')
st.markdown("#### It uses Deep Learning Method to find Similar Looking Product .")
st.markdown('#### It uses CNN model to find embedding for product images and then find the nearest neighbour using KNN .')
st.markdown('#### The CNN model used is EfficientNet-B3 with ArcFace Loss .')
st.text('')
st.text('')

base_path="test_images/"


default_value_goes_here="https://www.amazon.in/QONETIC-Rotating-Projector-Changing-Bedroom/dp/B08G1BMJ12/ref=sr_1_3_sspa?dchild=1&keywords=night+light&qid=1626663205&sr=8-3-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzSkRaUVVRMEtKSTlNJmVuY3J5cHRlZElkPUEwMTMwNzk5Mk1SR005SVNKRU9BNCZlbmNyeXB0ZWRBZElkPUEwODg3ODM4MVpKSDc2ME81QTQ2MiZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU="
user_input = st.text_input("Enter the Product Link", default_value_goes_here)

def output_prediction():
    df = pd.read_pickle("prediction.pkl")
    col1, col2, col3,col4 = st.beta_columns([2, 2, 2,2])
    with col1:
        st.markdown('### Product')

    with col2:
        st.markdown('### Name')

    with col3:
        st.markdown('### Price')

    with col4:
        st.markdown('### URL')

    st.markdown('### Given Product')
    html_string = "<br><br>"

    st.markdown(html_string, unsafe_allow_html=True)
    image = Image.open("test_images/" + df['image'][0])

    col1, col2, col3,col4 = st.beta_columns([2, 2, 2,2])
    with col1:
        st.image(image, width=200)

    with col2:
        st.markdown("### "+df['title'][0])

    with col3:
        st.markdown("### "+df['price'][0])

    with col4:
        st.markdown("### " + df['url'][0])

    st.markdown('### Recommended Similar Products')

    similar_product_index = df['image_predictions'][0]
    for i in similar_product_index:

        col1, col2, col3,col4 = st.beta_columns([2, 2, 2,2])
        with col1:
            st.image("test_images/" + df['image'][i], width=200)

        with col2:
            st.markdown("### " + df['title'][i])

        with col3:
            st.markdown("### " + df['price'][i])

        with col4:
            st.markdown("### " + df['url'][i])




def display_image(image,col1, col2,caption):
    with col1:
        st.write("")

    with col2:
        gif_runner=st.image(image,caption=caption)  # ,width=200,use_column_width=True)

    return gif_runner

if user_input :

    download_from_drive()
    if st.button('Search'):
        col1, col2 = st.beta_columns([1, 5])
        gif_runner = display_image("downloading.gif", col1, col2, "Downloading data")


        print(user_input, "here")


        create_dataset(user_input)
        gif_runner.empty()
        gif_runner = display_image("processing.gif",col1, col2,"Prccessing Data")
        predict()
        gif_runner.empty()

        output_prediction()


