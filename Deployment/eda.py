import streamlit as st
import pandas as pd
from PIL import Image

# Buat judul (page title) dipaling atas tab browser
st.set_page_config(
    page_title= 'Object klasifikasi Tanks atau Cars',
    layout="wide",
    initial_sidebar_state="auto"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

def runEDA():
    st.title('Object deteksi gambar Tank atau Cars (mobil) menggunakan Computer Vision CNN dan metrics f1-score')
    # buat deskripsi
    st.write('##### GC-7')
    st.write('##### Nama : Raden Rian Girianom')
    st.write('##### Batch : RMT-028')
    st.write('##### Source Dataset : [Click](https://www.kaggle.com/datasets/gatewayadam/cars-and-tanks-image-classification)')
    st.write('##### Projek ini bertujuan mengklasifikasi object deteksi gambar Tanks atau Cars menggunakan Computer Vision CNN dan metrics f1-score')

    st.markdown('---') # membuat garis pemisah ---
    st.subheader('Tahap EDA : Exploratory Data Analysis')

    # Buat EDA 1
    st.write('#### EDA 1 = 5 Image masing-masing class')
    st.image("EDA-1.jpg", caption="EDA-1", use_column_width=True)
    st.pyplot()

    # Buat EDA 2
    st.write('#### EDA 2 = Distribusi training-set')
    st.image("EDA-2.jpg", caption="EDA-1", use_column_width=True)
    st.pyplot()

    # Buat EDA 3
    st.write('#### EDA 2 = Distribusi testing-set')
    st.image("EDA-3.jpg", caption="EDA-1", use_column_width=True)
    st.pyplot()

if __name__ == '__main__':
    runEDA()