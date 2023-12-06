import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import plotly.graph_objs as go
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objs as go

def app() :
    
    st.title('HASIL METRIC PENGUJIAN RANDOM FOREST DENGAN RANDOM OVERSAMPLING DAN RANDOM FOREST TANPA RANDOM OVER SAMPLING')

    # Load Data Test
    x_test_a1= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_a1.pickle", "rb"))
    x_test_s1= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_s1.pickle", "rb"))
    x_test_a2= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_a2.pickle", "rb"))
    x_test_s2= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_s2.pickle", "rb"))
    x_test_a3= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_a3.pickle", "rb"))
    x_test_s3= pickle.load(open("D:\\skripsi\\sidang\\code1\\x_test\\x_test_s3.pickle", "rb"))
    y_test_a1= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_a1.pickle", "rb"))
    y_test_s1= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_s1.pickle", "rb"))
    y_test_a2= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_a2.pickle", "rb"))
    y_test_s2= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_s2.pickle", "rb"))
    y_test_a3= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_a3.pickle", "rb"))
    y_test_s3= pickle.load(open("D:\\skripsi\\sidang\\code1\\y_test\\y_test_s3.pickle", "rb"))
    
    # Random Forest
    # Load Model
    model_a1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a1.pickle','rb'))
    model_a2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a2.pickle','rb'))
    model_a3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a3.pickle','rb'))
    model_s1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s1.pickle','rb'))
    model_s2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s2.pickle','rb'))
    model_s3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s3.pickle','rb'))

    # Load kamus vocab dan membuat objek TfidfVectorizer dengan kamus vocab
    saved_vocabulary_a1 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_a1.pickle", 'rb'))
    tfidf_a1 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a1)

    saved_vocabulary_a2 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_a2.pickle", 'rb'))
    tfidf_a2 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a2)
    
    saved_vocabulary_a3 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_a3.pickle", 'rb'))
    tfidf_a3 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a3)

    saved_vocabulary_s1 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_s1.pickle", 'rb'))
    tfidf_s1 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s1)

    saved_vocabulary_s2 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_s2.pickle", 'rb'))
    tfidf_s2 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s2)

    saved_vocabulary_s3 = pickle.load(open("D:\\skripsi\\streamlit\model\\TFIDF RF\\TFIDF_s3.pickle", 'rb'))  
    tfidf_s3 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s3)

    #Mengubah data test menjadi vektor TFIDF
    new_vect_a1= tfidf_a1.fit_transform(x_test_a1)
    new_vect_a2= tfidf_a2.fit_transform(x_test_a2)
    new_vect_a3= tfidf_a3.fit_transform(x_test_a3)
    new_vect_s1= tfidf_s1.fit_transform(x_test_s1)
    new_vect_s2= tfidf_s2.fit_transform(x_test_s2)
    new_vect_s3= tfidf_s3.fit_transform(x_test_s3)

    # melakukan prediksi dengan model 
    result_a1 = model_a1.predict(new_vect_a1)  
    result_a2 = model_a2.predict(new_vect_a2)
    result_a3 = model_a3.predict(new_vect_a3)
    result_s1 = model_s1.predict(new_vect_s1)
    result_s2 = model_s2.predict(new_vect_s2) 
    result_s3 = model_s3.predict(new_vect_s3)  

    # Menghitung skor geometric mean untuk setiap model 
    gmean_a1 = geometric_mean_score(y_test_a1, result_a1)
    gmean_a2 = geometric_mean_score(y_test_a2, result_a2)
    gmean_a3 = geometric_mean_score(y_test_a3, result_a3)
    gmean_s1 = geometric_mean_score(y_test_s1, result_s1)
    gmean_s2 = geometric_mean_score(y_test_s2, result_s2)
    gmean_s3 = geometric_mean_score(y_test_s3, result_s3)

    # Menghitung skor f1 score untuk setiap model
    f1_score_a1 = metrics.f1_score(y_test_a1, result_a1, average='macro')
    f1_score_a2 = metrics.f1_score(y_test_a2, result_a2, average='macro')
    f1_score_a3 = metrics.f1_score(y_test_a3, result_a3, average='macro')
    f1_score_s1 = metrics.f1_score(y_test_s1, result_s1, average='macro')
    f1_score_s2 = metrics.f1_score(y_test_s2, result_s2, average='macro')
    f1_score_s3 = metrics.f1_score(y_test_s3, result_s3, average='macro')
    
    # Menghitung skor balanced accuracy untuk setiap model
    balanced_accuracy_a1 = balanced_accuracy_score(y_test_a1, result_a1)
    balanced_accuracy_a2 = balanced_accuracy_score(y_test_a2, result_a2)
    balanced_accuracy_a3 = balanced_accuracy_score(y_test_a3, result_a3)
    balanced_accuracy_s1 = balanced_accuracy_score(y_test_s1, result_s1)
    balanced_accuracy_s2 = balanced_accuracy_score(y_test_s2, result_s2)
    balanced_accuracy_s3 = balanced_accuracy_score(y_test_s3, result_s3)

    
    #RANDOM FOREST + RANDOM OVER SAMPLING  

    # Load Model
    model_a1_ROS = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a1_ROS.pickle','rb'))
    model_a2_ROS = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a2_ROS.pickle','rb'))
    model_a3_ROS= pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a3_ROS.pickle','rb'))
    model_s1_ROS = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s1_ROS.pickle','rb'))
    model_s2_ROS= pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s2_ROS.pickle','rb'))
    model_s3_ROS = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s3_ROS.pickle','rb'))

    # Load kamus vocab dan membuat objek tfidf dengan kamus vocab
    saved_vocabulary_a1_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a1_ROS.pickle", 'rb'))
    tfidf_a1_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a1_ROS)

    saved_vocabulary_a2_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a2_ROS.pickle", 'rb'))
    tfidf_a2_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a2_ROS)

    saved_vocabulary_a3_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a3_ROS.pickle", 'rb'))
    tfidf_a3_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a3_ROS)

    saved_vocabulary_s1_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s1_ROS.pickle", 'rb'))
    tfidf_s1_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s1_ROS)

    saved_vocabulary_s2_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s2_ROS.pickle", 'rb'))
    tfidf_s2_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s2_ROS)

    saved_vocabulary_s3_ROS = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s3_ROS.pickle", 'rb'))  
    tfidf_s3_ROS = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s3_ROS)

    # mengubah data test menjadi vektor tfidf 
    new_vect_a1_ROS= tfidf_a1_ROS.fit_transform(x_test_a1)
    new_vect_a2_ROS= tfidf_a2_ROS.fit_transform(x_test_a2)
    new_vect_a3_ROS= tfidf_a3_ROS.fit_transform(x_test_a3)
    new_vect_s1_ROS= tfidf_s1_ROS.fit_transform(x_test_s1)
    new_vect_s2_ROS= tfidf_s2_ROS.fit_transform(x_test_s2)
    new_vect_s3_ROS= tfidf_s3_ROS.fit_transform(x_test_s3)
    
    # melakukan prediksi dengan model
    result_a1_ROS = model_a1_ROS.predict(new_vect_a1_ROS)  
    result_a2_ROS = model_a2_ROS.predict(new_vect_a2_ROS)
    result_a3_ROS = model_a3_ROS.predict(new_vect_a3_ROS)
    result_s1_ROS = model_s1_ROS.predict(new_vect_s1_ROS)
    result_s2_ROS = model_s2_ROS.predict(new_vect_s2_ROS) 
    result_s3_ROS = model_s3_ROS.predict(new_vect_s3_ROS)  
    
    # Menghitung skor geometric mean untuk setiap model 
    gmean_a1_ROS = geometric_mean_score(y_test_a1, result_a1_ROS)
    gmean_a2_ROS = geometric_mean_score(y_test_a2, result_a2_ROS)
    gmean_a3_ROS = geometric_mean_score(y_test_a3, result_a3_ROS)
    gmean_s1_ROS = geometric_mean_score(y_test_s1, result_s1_ROS)
    gmean_s2_ROS = geometric_mean_score(y_test_s2, result_s2_ROS)
    gmean_s3_ROS = geometric_mean_score(y_test_s3, result_s3_ROS)

    # Menghitung skor F1 Score untuk setiap model 
    f1_score_a1_ROS = metrics.f1_score(y_test_a1, result_a1_ROS, average='macro')
    f1_score_a2_ROS= metrics.f1_score(y_test_a2, result_a2_ROS, average='macro')
    f1_score_a3_ROS = metrics.f1_score(y_test_a3, result_a3_ROS, average='macro')
    f1_score_s1_ROS = metrics.f1_score(y_test_s1, result_s1_ROS, average='macro')
    f1_score_s2_ROS = metrics.f1_score(y_test_s2, result_s2_ROS, average='macro')
    f1_score_s3_ROS = metrics.f1_score(y_test_s3, result_s3_ROS, average='macro')
    
    # Menghitung skor balanced accuracy untuk setiap model 
    balanced_accuracy_a1_ROS = balanced_accuracy_score(y_test_a1, result_a1_ROS)
    balanced_accuracy_a2_ROS = balanced_accuracy_score(y_test_a2, result_a2_ROS)
    balanced_accuracy_a3_ROS = balanced_accuracy_score(y_test_a3, result_a3_ROS)
    balanced_accuracy_s1_ROS = balanced_accuracy_score(y_test_s1, result_s1_ROS)
    balanced_accuracy_s2_ROS = balanced_accuracy_score(y_test_s2, result_s2_ROS)
    balanced_accuracy_s3_ROS = balanced_accuracy_score(y_test_s3, result_s3_ROS)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row} </style>', unsafe_allow_html=True)
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:12px; justify-content: center;}</style>', unsafe_allow_html=True)
    
    choose=st.radio("PILIH MODEL : ",('Aspek Atraksi', 'Aspek Amenitas', 'Aspek Aksesibilitas', 'Sentimen Atraksi', 'Sentimen Amenitas', 'Sentimen Aksesibilitas'))

    st.markdown("""---""")
    # st.markdown("<hr style='border-top: 2px solid black;'>", unsafe_allow_html=True)
    # Menampilkan hasil berdasarkan model yang dipilih
    if choose == 'Aspek Atraksi':
        # Tampilkan plot atau informasi untuk model a1
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy A1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_a1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_a1 = [balanced_accuracy_a1 * 100, balanced_accuracy_a1_ROS * 100]

        # Membuat plot batang
        data_ba_a1 = [go.Bar(
            x=labels_ba_a1,
            y=values_ba_a1,
            text=[f'{value:.2f}%' for value in values_ba_a1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_a1 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_a1 = go.Figure(data=data_ba_a1, layout=layout_ba_a1)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_a1, use_container_width=True)



        # Grafik F1 Score A1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_a1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_a1 = [f1_score_a1 * 100, f1_score_a1_ROS * 100]

        # Membuat plot batang
        data_f1_a1 = [go.Bar(
            x=labels_f1_a1,
            y=values_f1_a1,
            text=[f'{value:.2f}%' for value in values_f1_a1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_a1 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_a1 = go.Figure(data=data_f1_a1, layout=layout_f1_a1)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_a1, use_container_width=True)



        # Grafik G-Mean A1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_a1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_a1 = [gmean_a1 * 100, gmean_a1_ROS * 100]

        # Membuat plot batang
        data_gm_a1 = [go.Bar(
            x=labels_gm_a1,
            y=values_gm_a1,
            text=[f'{value:.2f}%' for value in values_gm_a1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_a1 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_a1 = go.Figure(data=data_gm_a1, layout=layout_gm_a1)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_a1, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_a1 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_a1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_a1_diff = (balanced_accuracy_a1_ROS - balanced_accuracy_a1)
        abs_balanced_accuracy_a1_diff = abs(balanced_accuracy_a1_diff)
    
        # Tampilkan nilai `balanced_accuracy_a1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_a1_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a1_diff*100, 3)}% {up_arrow_ba_a1}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_a1_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a1_diff*100, 3)}% {down_arrow_ba_a1}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a1_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_a1_diff = (f1_score_a1_ROS - f1_score_a1)
        abs_f1_score_a1_diff = abs(f1_score_a1_diff)

        up_arrow_f1_a1 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_a1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_a1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_a1_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a1_diff*100, 3)}% {up_arrow_f1_a1}</p>", unsafe_allow_html=True)
        elif f1_score_a1_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a1_diff*100, 3)}% {down_arrow_f1_a1}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a1_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_a1_diff = (gmean_a1_ROS - gmean_a1)
        abs_gmean_a1_diff = abs(gmean_a1_diff)

        up_arrow_gm_a1 = "<span style='color:#B03A2E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_a1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_a1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_a1_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a1_diff*100, 3)}% {up_arrow_gm_a1}</p>", unsafe_allow_html=True)
        elif gmean_a1_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a1_diff*100, 3)}% {down_arrow_gm_a1}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a1_diff*100, 3)}%</p>", unsafe_allow_html=True)

    elif choose == 'Aspek Amenitas':
                # Tampilkan plot atau informasi untuk model a2
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy a2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_a2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_a2 = [balanced_accuracy_a2 * 100, balanced_accuracy_a2_ROS * 100]

        # Membuat plot batang
        data_ba_a2 = [go.Bar(
            x=labels_ba_a2,
            y=values_ba_a2,
            text=[f'{value:.2f}%' for value in values_ba_a2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_a2 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_a2 = go.Figure(data=data_ba_a2, layout=layout_ba_a2)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_a2, use_container_width=True)



        # Grafik F1 Score a2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_a2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_a2 = [f1_score_a2 * 100, f1_score_a2_ROS * 100]

        # Membuat plot batang
        data_f1_a2 = [go.Bar(
            x=labels_f1_a2,
            y=values_f1_a2,
            text=[f'{value:.2f}%' for value in values_f1_a2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_a2 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_a2 = go.Figure(data=data_f1_a2, layout=layout_f1_a2)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_a2, use_container_width=True)



        # Grafik G-Mean a2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_a2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_a2 = [gmean_a2 * 100, gmean_a2_ROS * 100]

        # Membuat plot batang
        data_gm_a2 = [go.Bar(
            x=labels_gm_a2,
            y=values_gm_a2,
            text=[f'{value:.2f}%' for value in values_gm_a2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_a2 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_a2 = go.Figure(data=data_gm_a2, layout=layout_gm_a2)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_a2, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_a2 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_a2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_a2_diff = (balanced_accuracy_a2_ROS - balanced_accuracy_a2)
        abs_balanced_accuracy_a2_diff = abs(balanced_accuracy_a2_diff)
    
        # Tampilkan nilai `balanced_accuracy_a2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_a2_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a2_diff*100, 3)}% {up_arrow_ba_a2}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_a2_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a2_diff*100, 3)}% {down_arrow_ba_a2}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a2_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_a2_diff = (f1_score_a2_ROS - f1_score_a2)
        abs_f1_score_a2_diff = abs(f1_score_a2_diff)

        up_arrow_f1_a2 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_a2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_a2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_a2_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a2_diff*100, 3)}% {up_arrow_f1_a2}</p>", unsafe_allow_html=True)
        elif f1_score_a2_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a2_diff*100, 3)}% {down_arrow_f1_a2}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a2_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_a2_diff = (gmean_a2_ROS - gmean_a2)
        abs_gmean_a2_diff = abs(gmean_a2_diff)

        up_arrow_gm_a2 = "<span style='color:#B03A2E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_a2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_a2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_a2_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a2_diff*100, 3)}% {up_arrow_gm_a2}</p>", unsafe_allow_html=True)
        elif gmean_a2_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a2_diff*100, 3)}% {down_arrow_gm_a2}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a2_diff*100, 3)}%</p>", unsafe_allow_html=True)

    elif choose == 'Aspek Aksesibilitas':
                # Tampilkan plot atau informasi untuk model a3
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy a3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_a3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_a3 = [balanced_accuracy_a3 * 100, balanced_accuracy_a3_ROS * 100]

        # Membuat plot batang
        data_ba_a3 = [go.Bar(
            x=labels_ba_a3,
            y=values_ba_a3,
            text=[f'{value:.2f}%' for value in values_ba_a3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_a3 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_a3 = go.Figure(data=data_ba_a3, layout=layout_ba_a3)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_a3, use_container_width=True)



        # Grafik F1 Score a3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_a3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_a3 = [f1_score_a3 * 100, f1_score_a3_ROS * 100]

        # Membuat plot batang
        data_f1_a3 = [go.Bar(
            x=labels_f1_a3,
            y=values_f1_a3,
            text=[f'{value:.2f}%' for value in values_f1_a3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_a3 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_a3 = go.Figure(data=data_f1_a3, layout=layout_f1_a3)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_a3, use_container_width=True)



        # Grafik G-Mean a3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_a3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_a3 = [gmean_a3 * 100, gmean_a3_ROS * 100]

        # Membuat plot batang
        data_gm_a3 = [go.Bar(
            x=labels_gm_a3,
            y=values_gm_a3,
            text=[f'{value:.2f}%' for value in values_gm_a3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_a3 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_a3 = go.Figure(data=data_gm_a3, layout=layout_gm_a3)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_a3, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_a3 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_a3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_a3_diff = (balanced_accuracy_a3_ROS - balanced_accuracy_a3)
        abs_balanced_accuracy_a3_diff = abs(balanced_accuracy_a3_diff)
    
        # Tampilkan nilai `balanced_accuracy_a3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_a3_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a3_diff*100, 3)}% {up_arrow_ba_a3}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_a3_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a3_diff*100, 3)}% {down_arrow_ba_a3}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_a3_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_a3_diff = (f1_score_a3_ROS - f1_score_a3)
        abs_f1_score_a3_diff = abs(f1_score_a3_diff)

        up_arrow_f1_a3 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_a3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_a3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_a3_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a3_diff*100, 3)}% {up_arrow_f1_a3}</p>", unsafe_allow_html=True)
        elif f1_score_a3_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a3_diff*100, 3)}% {down_arrow_f1_a3}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_a3_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_a3_diff = (gmean_a3_ROS - gmean_a3)
        abs_gmean_a3_diff = abs(gmean_a3_diff)

        up_arrow_gm_a3 = "<span style='color:#B03a3E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_a3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_a3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_a3_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a3_diff*100, 3)}% {up_arrow_gm_a3}</p>", unsafe_allow_html=True)
        elif gmean_a3_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a3_diff*100, 3)}% {down_arrow_gm_a3}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_a3_diff*100, 3)}%</p>", unsafe_allow_html=True)

    elif choose == 'Sentimen Atraksi':
                # Tampilkan plot atau informasi untuk model s1
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy s1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_s1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_s1 = [balanced_accuracy_s1 * 100, balanced_accuracy_s1_ROS * 100]

        # Membuat plot batang
        data_ba_s1 = [go.Bar(
            x=labels_ba_s1,
            y=values_ba_s1,
            text=[f'{value:.2f}%' for value in values_ba_s1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_s1 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_s1 = go.Figure(data=data_ba_s1, layout=layout_ba_s1)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_s1, use_container_width=True)



        # Grafik F1 Score s1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_s1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_s1 = [f1_score_s1 * 100, f1_score_s1_ROS * 100]

        # Membuat plot batang
        data_f1_s1 = [go.Bar(
            x=labels_f1_s1,
            y=values_f1_s1,
            text=[f'{value:.2f}%' for value in values_f1_s1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_s1 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_s1 = go.Figure(data=data_f1_s1, layout=layout_f1_s1)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_s1, use_container_width=True)



        # Grafik G-Mean s1
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_s1 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_s1 = [gmean_s1 * 100, gmean_s1_ROS * 100]

        # Membuat plot batang
        data_gm_s1 = [go.Bar(
            x=labels_gm_s1,
            y=values_gm_s1,
            text=[f'{value:.2f}%' for value in values_gm_s1],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_s1 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_s1 = go.Figure(data=data_gm_s1, layout=layout_gm_s1)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_s1, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_s1 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_s1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_s1_diff = (balanced_accuracy_s1_ROS - balanced_accuracy_s1)
        abs_balanced_accuracy_s1_diff = abs(balanced_accuracy_s1_diff)
    
        # Tampilkan nilai `balanced_accuracy_s1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_s1_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s1_diff*100, 3)}% {up_arrow_ba_s1}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_s1_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s1_diff*100, 3)}% {down_arrow_ba_s1}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s1_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_s1_diff = (f1_score_s1_ROS - f1_score_s1)
        abs_f1_score_s1_diff = abs(f1_score_s1_diff)

        up_arrow_f1_s1 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_s1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_s1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_s1_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s1_diff*100, 3)}% {up_arrow_f1_s1}</p>", unsafe_allow_html=True)
        elif f1_score_s1_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s1_diff*100, 3)}% {down_arrow_f1_s1}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s1_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_s1_diff = (gmean_s1_ROS - gmean_s1)
        abs_gmean_s1_diff = abs(gmean_s1_diff)

        up_arrow_gm_s1 = "<span style='color:#B03A2E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_s1 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_s1_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_s1_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s1_diff*100, 3)}% {up_arrow_gm_s1}</p>", unsafe_allow_html=True)
        elif gmean_s1_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s1_diff*100, 3)}% {down_arrow_gm_s1}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s1_diff*100, 3)}%</p>", unsafe_allow_html=True)

    elif choose == 'Sentimen Amenitas':
                # Tampilkan plot atau informasi untuk model s2
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy s2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_s2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_s2 = [balanced_accuracy_s2 * 100, balanced_accuracy_s2_ROS * 100]

        # Membuat plot batang
        data_ba_s2 = [go.Bar(
            x=labels_ba_s2,
            y=values_ba_s2,
            text=[f'{value:.2f}%' for value in values_ba_s2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_s2 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_s2 = go.Figure(data=data_ba_s2, layout=layout_ba_s2)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_s2, use_container_width=True)



        # Grafik F1 Score s2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_s2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_s2 = [f1_score_s2 * 100, f1_score_s2_ROS * 100]

        # Membuat plot batang
        data_f1_s2 = [go.Bar(
            x=labels_f1_s2,
            y=values_f1_s2,
            text=[f'{value:.2f}%' for value in values_f1_s2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_s2 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_s2 = go.Figure(data=data_f1_s2, layout=layout_f1_s2)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_s2, use_container_width=True)



        # Grafik G-Mean s2
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_s2 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_s2 = [gmean_s2 * 100, gmean_s2_ROS * 100]

        # Membuat plot batang
        data_gm_s2 = [go.Bar(
            x=labels_gm_s2,
            y=values_gm_s2,
            text=[f'{value:.2f}%' for value in values_gm_s2],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_s2 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_s2 = go.Figure(data=data_gm_s2, layout=layout_gm_s2)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_s2, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_s2 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_s2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_s2_diff = (balanced_accuracy_s2_ROS - balanced_accuracy_s2)
        abs_balanced_accuracy_s2_diff = abs(balanced_accuracy_s2_diff)
    
        # Tampilkan nilai `balanced_accuracy_s2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_s2_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s2_diff*100, 3)}% {up_arrow_ba_s2}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_s2_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s2_diff*100, 3)}% {down_arrow_ba_s2}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s2_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_s2_diff = (f1_score_s2_ROS - f1_score_s2)
        abs_f1_score_s2_diff = abs(f1_score_s2_diff)

        up_arrow_f1_s2 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_s2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_s2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_s2_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s2_diff*100, 3)}% {up_arrow_f1_s2}</p>", unsafe_allow_html=True)
        elif f1_score_s2_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s2_diff*100, 3)}% {down_arrow_f1_s2}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s2_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_s2_diff = (gmean_s2_ROS - gmean_s2)
        abs_gmean_s2_diff = abs(gmean_s2_diff)

        up_arrow_gm_s2 = "<span style='color:#B03A2E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_s2 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_s2_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_s2_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s2_diff*100, 3)}% {up_arrow_gm_s2}</p>", unsafe_allow_html=True)
        elif gmean_s2_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s2_diff*100, 3)}% {down_arrow_gm_s2}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s2_diff*100, 3)}%</p>", unsafe_allow_html=True)
 
    elif choose == 'Sentimen Aksesibilitas':
                # Tampilkan plot atau informasi untuk model s3
        # st.markdown("<h2 style='text-align: center;'>Aspek Atraksi (Derajat Ketidakseimbangan : Ringan)</h2>", unsafe_allow_html=True)
        col1, col2, col3= st.columns(3)
        
        # Grafik Balanced accuracy s3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_ba_s3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_ba_s3 = [balanced_accuracy_s3 * 100, balanced_accuracy_s3_ROS * 100]

        # Membuat plot batang
        data_ba_s3 = [go.Bar(
            x=labels_ba_s3,
            y=values_ba_s3,
            text=[f'{value:.2f}%' for value in values_ba_s3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='midnightblue'), # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
            
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_ba_s3 = go.Layout(
            title=dict(text='Perbandingan Balanced Accuracy', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),  # Set the font size of x-axis label to 14
            yaxis=dict(title='Balanced Accuracy (%)', titlefont=dict(size=15)),  # Set the font size of y-axis label to 14
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_ba_s3 = go.Figure(data=data_ba_s3, layout=layout_ba_s3)

        # Menampilkan plot menggunakan Streamlit
        col1.plotly_chart(fig_ba_s3, use_container_width=True)



        # Grafik F1 Score s3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_f1_s3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_f1_s3 = [f1_score_s3 * 100, f1_score_s3_ROS * 100]

        # Membuat plot batang
        data_f1_s3 = [go.Bar(
            x=labels_f1_s3,
            y=values_f1_s3,
            text=[f'{value:.2f}%' for value in values_f1_s3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#006400'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_f1_s3 = go.Layout(
            title=dict(text='Perbandingan F1 Score', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='F1 Score (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500,  # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_f1_s3 = go.Figure(data=data_f1_s3, layout=layout_f1_s3)

        # Menampilkan plot menggunakan Streamlit
        col2.plotly_chart(fig_f1_s3, use_container_width=True)



        # Grafik G-Mean s3
        # Nilai-nilai yang akan ditampilkan pada grafik batang
        labels_gm_s3 = ['Aspek Atraksi RF', 'Aspek Atraksi RF + ROS']
        values_gm_s3 = [gmean_s3 * 100, gmean_s3_ROS * 100]

        # Membuat plot batang
        data_gm_s3 = [go.Bar(
            x=labels_gm_s3,
            y=values_gm_s3,
            text=[f'{value:.2f}%' for value in values_gm_s3],  # Menambahkan label persen pada setiap batang
            textposition='auto',  # Menentukan posisi label pada batang
            marker=dict(color='#B03A2E'),  # Mengubah warna batang menjadi merah
            textfont=dict(size=18),
            width=[0.8,0.8]
        )]

        # Menentukan tata letak grafik dengan opsi autosize dan height
        layout_gm_s3 = go.Layout(
            title=dict(text='Perbandingan G-Mean', x=0.5),
            titlefont=dict(size=18),  # Set the font size to 20
            xaxis=dict(title='Metode', titlefont=dict(size=15)),
            yaxis=dict(title='G-Mean (%)', titlefont=dict(size=15)),
            autosize=True,  # Mengatur ukuran plot secara otomatis
            height=500, # Mengatur tinggi plot
            margin=dict(l=50, r=50, t=50, b=200)
        )

        # Membuat figure dengan data dan layout
        fig_gm_s3 = go.Figure(data=data_gm_s3, layout=layout_gm_s3)

        # Menampilkan plot menggunakan Streamlit
        col3.plotly_chart(fig_gm_s3, use_container_width=True)

        # st.markdown("""---""")
        # st.markdown("<h2 style='text-align: center;'>F1-Score</h2>", unsafe_allow_html=True)
        #DAMPAK ROS

        col1,col2,col3 = st.columns(3)
        up_arrow_ba_s3 = "<span style='color:midnightblue; font-size:1.2em;'>&#9650;</span>"
        down_arrow_ba_s3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        
        col1_margin_top = "-150px"
        # calculate difference between balanced accuracy model RF dan model RF+ROS
        balanced_accuracy_s3_diff = (balanced_accuracy_s3_ROS - balanced_accuracy_s3)
        abs_balanced_accuracy_s3_diff = abs(balanced_accuracy_s3_diff)
    
        # Tampilkan nilai `balanced_accuracy_s3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if balanced_accuracy_s3_diff > 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s3_diff*100, 3)}% {up_arrow_ba_s3}</p>", unsafe_allow_html=True)
        elif balanced_accuracy_s3_diff < 0:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s3_diff*100, 3)}% {down_arrow_ba_s3}</p>", unsafe_allow_html=True)

        else:
            with col1:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col1_margin_top};'>{round(abs_balanced_accuracy_s3_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col2_margin_top = "-150px"
        # calculate difference between F1-Score model RF dan model RF+ROS
        f1_score_s3_diff = (f1_score_s3_ROS - f1_score_s3)
        abs_f1_score_s3_diff = abs(f1_score_s3_diff)

        up_arrow_f1_s3 = "<span style='color:#006400; font-size:1.2em;'>&#9650;</span>"
        down_arrow_f1_s3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `f1_score_s3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if f1_score_s3_diff > 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s3_diff*100, 3)}% {up_arrow_f1_s3}</p>", unsafe_allow_html=True)
        elif f1_score_s3_diff < 0:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s3_diff*100, 3)}% {down_arrow_f1_s3}</p>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col2_margin_top};'>{round(abs_f1_score_s3_diff*100, 3)}%</p>", unsafe_allow_html=True)


        col3_margin_top = "-150px"
        # calculate difference between G-Mean model RF dan model RF+ROS
        gmean_s3_diff = (gmean_s3_ROS - gmean_s3)
        abs_gmean_s3_diff = abs(gmean_s3_diff)

        up_arrow_gm_s3 = "<span style='color:#B03A2E; font-size:1.2em;'>&#9650;</span>"
        down_arrow_gm_s3 = "<span style='color:red; font-size:1.2em;'>&#9660;</span>"
        # Tampilkan nilai `gmean_s3_diff` dan tanda panah yang menunjukkan apakah meningkat atau menurun.
        if gmean_s3_diff > 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s3_diff*100, 3)}% {up_arrow_gm_s3}</p>", unsafe_allow_html=True)
        elif gmean_s3_diff < 0:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s3_diff*100, 3)}% {down_arrow_gm_s3}</p>", unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"<center><p style='font-size:40px; margin-top: {col3_margin_top};'>{round(abs_gmean_s3_diff*100, 3)}%</p>", unsafe_allow_html=True)
     