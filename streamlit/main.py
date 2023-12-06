import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import streamlit as st
import pandas as pd
import numpy as np
import io
import nltk, re, string, ast
from nltk.tokenize import word_tokenize
from string import punctuation
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from view import klasifikasiText,klasifikasiFile,visualisasi,pengujian
import utils as uts
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
uts.inject_custom_css()
uts.navbar_component()

def navigation():
    route = uts.get_current_route()
    if route == "klasifikasiText":
        klasifikasiText.app()
    elif route == "klasifikasiFile":
        klasifikasiFile.app()
    elif route == "visualisasi":
        visualisasi.app()
    elif route == "pengujian":
        pengujian.app()
navigation()

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://lamongantourism.com/wp-content/uploads/2018/11/goa-maharani-2.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url() 

    
