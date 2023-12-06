import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    def visualisasi_data_aspek(df):
        # ambil 3 kolom terakhir dari dataframe
        aspects = df.columns[-3:]

        # buat dictionary untuk simpan data visualisasi
        data = {'aspek': [], 'exist': [], 'not_exist': []}

        for col_name in aspects:
            # hitung jumlah data yang berlabel positif dan negatif
            exist_count = len(df[(df[col_name] == 'positif') | (df[col_name] == 'negatif')])
            # hitung jumlah data tidak berlabel
            not_exist_count = len(df) - exist_count
            # simpan nama aspek ke dictionary
            data['aspek'].append(col_name)
            # simpan jumlah data exist ke dictionary
            data['exist'].append(exist_count)
            # simpan jumlah data not exist ke dictionary
            data['not_exist'].append(not_exist_count)
        fig = px.bar(data_frame=data, x='aspek', y=['exist', 'not_exist'], barmode='group', title='Grafik data hasil pelabelan aspek') # Membuat grafik batang dengan menggunakan data yang telah dikumpulkan
        fig.update_layout(
            xaxis=dict(title='Aspek'), # Menambahkan label sumbu x
            yaxis=dict(title='Jumlah data'),  # Menambahkan label sumbu y
            legend=dict(title='Label'), # Menambahkan legenda
            plot_bgcolor='white', # Mengatur warna latar plot
            font=dict(color='black') # Mengatur warna teks
        )
        return fig

    def visualisasi_data_sentimen(df):
        # ambil 3 kolom terakhir dari dataframe
        aspects = df.columns[-3:]
        # buat dictionary untuk simpan data visualisasi
        data = {'aspek': [], 'sentimen': [], 'jumlah_data': []}
        for col_name in aspects:
            # hitung jumlah data yang berlabel positif 
            pos_count = len(df[df[col_name] == 'positif'])
            # hitung jumlah data yang berlabel negatif
            neg_count = len(df[df[col_name] == 'negatif'])
            # simpan nama aspek (positif dan negatif)
            data['aspek'] += [col_name, col_name]
            # simpan label sentimen (positif negatif)
            data['sentimen'] += ['positif', 'negatif']
            # simpan jumlah data positif dan negatif
            data['jumlah_data'] += [pos_count, neg_count]

        # Membuat grafik batang dengan menggunakan data yang telah dikumpulkan
        # Menggunakan 'aspek' sebagai sumbu x, 'jumlah_data' sebagai sumbu y, dan 'sentimen' sebagai warna batang
        # Mengatur judul grafik
        fig = px.bar(data_frame=data, x='aspek', y='jumlah_data', color='sentimen', barmode='group', title='Grafik data hasil pelabelan sentimen')
        fig.update_layout(
            xaxis=dict(title='Aspek'),
            yaxis=dict(title='Jumlah data'),
            legend=dict(title='Sentimen'),
            plot_bgcolor='white',
            font=dict(color='black')
        )
        return fig # Mengembalikan objek grafik batang

    st.title('VISUALISASI DATA ASPEK DAN SENTIMEN ULASAN PENGUNJUNG OBJEK WISATA KABUPATEN LAMONGAN DI GOOGLE MAPS')
    uploaded_file = st.file_uploader("Pilih file", type=['csv'])
    submit_file = st.button("Visualisasi")
    if submit_file:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            col1, col2 = st.columns(2)
            col1.plotly_chart(visualisasi_data_aspek(data),use_container_width=True)
            col2.plotly_chart(visualisasi_data_sentimen(data),use_container_width=True)