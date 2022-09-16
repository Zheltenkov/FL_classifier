import base64
import streamlit as st
from config.config import parse_args
from scripts.test import get_prediction

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


if __name__=='__main__':

    st.set_page_config(
         page_title="Flower Classifier App",
         page_icon="🌼",
         layout="centered",
         initial_sidebar_state="expanded")

    img_bgd = get_img_as_base64('./data/application/background.jpg')

    page_bg_img = (f"""
    <style>
    [data-testid="stAppViewContainer"]{{
    background-image: url("data:image/png;base64,{img_bgd}");
    background-size: cover;
    }}
    
    div.stButton > button:first-child {{
    background-color: #555555;
    height:50px; width:200px; border: none; font-size: 26px;
    color: rgb(255, 255, 255); 
    }}
    
    span.class="css-10trblm e16nr0p30"{{
    style="color:#ff6347";
    }} 
    
    </style>
    """)

    m = st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title('Flower classifier')
    st.subheader('Load flowers images')

    uploaded_file = st.file_uploader("",
                                     accept_multiple_files=True,
                                     help='Drop a picture of flower in drag zone')

    generate_pred = st.button('Predict')

    args = parse_args()

    if generate_pred:
        if len(uploaded_file) != 0:
            for image in uploaded_file:
                class_name = get_prediction(args, image, args.trained_model)
                st.title(f'Flower is - {class_name.upper()}!')
                st.image(image)
        else:
            st.title(f'There is nothing to predict....')