import streamlit as st
import pandas as pd
import nltk, re, string, ast
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Text Preprocessing
def app() :
    def cleansing(text):
        for i in text:
            if i in list(string.punctuation):
                text = text.replace(i, " ")
        text = re.sub(r"\d+", " ", text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def case_folding(text):
        text = text.lower()
        return text

    def normalisasi(texts):
        FILE_NORMALISASI_PATH = 'D:\\ainandiyah\\streamlit\\model\\normal.txt'

        with open(FILE_NORMALISASI_PATH, encoding="utf-8") as f:
            data_normalisasi = f.read()
        normalization_words = ast.literal_eval(data_normalisasi)

        finalText = []
        splitted_text = texts.split()
        for text in splitted_text:
            if text in normalization_words:
                finalText.append(normalization_words[text])
            else:
                finalText.append(text)
        return " ".join(finalText)

    def hapus_stopword(text):
        FILE_STOPWORDS_PATH = 'D:\\ainandiyah\\streamlit\\model\\stopwords.txt'
        
        with open(FILE_STOPWORDS_PATH, encoding="utf-8") as f:
            data_stopwords = f.read()
        stopwords = ast.literal_eval(data_stopwords)

        stopword_factory = stopwords

        sw_dict = ArrayDictionary(stopword_factory)
        temp = StopWordRemover(sw_dict)

        text = temp.remove(text)
        return text
    
    def stemming(text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
        return text

    def preprocessing_data(kalimat):
        kalimat = cleansing(kalimat)
        kalimat = case_folding(kalimat)
        kalimat = normalisasi(kalimat)
        kalimat = hapus_stopword(kalimat)
        kalimat = stemming(kalimat)
        return kalimat
    
    st.title('KLASIFIKASI FILE ULASAN PENGUNJUNG OBJEK WISATA KABUPATEN LAMONGAN DI GOOGLE MAPS')
    uploaded_file = st.file_uploader("Pilih file", type = ['csv'])
    ml_choice = st.selectbox("Pilih Metode", ["Random Forest","Random Forest (Random Over Sampling)"])
    submit_file = st.button("Analisis")
    if submit_file:
        if uploaded_file is not None :
            if ml_choice == "Random Forest" :
                # Read data yang di upload
                data = pd.read_csv(uploaded_file)

                #Load model RF
                model_a1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a1.pickle','rb'))
                model_a2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a2.pickle','rb'))
                model_a3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a3.pickle','rb'))
                model_s1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s1.pickle','rb'))
                model_s2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s2.pickle','rb'))
                model_s3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s3.pickle','rb'))

                #Load kamus vocab dan membuat objek TfidfVectorizer dengan kamus vocab
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
 
                #memuat ulasan dari data
                ulasan=data["ulasan"]

                #melakukan text preprocessing pada teks ulasan
                X_processed_text = [preprocessing_data(row) for row in ulasan]

                #Mengubah data baru menjadi vektor TFIDF
                new_vect_a1= tfidf_a1.fit_transform(X_processed_text)
                new_vect_a2= tfidf_a2.fit_transform(X_processed_text)
                new_vect_a3= tfidf_a3.fit_transform(X_processed_text)
                new_vect_s1= tfidf_s1.fit_transform(X_processed_text)
                new_vect_s2= tfidf_s2.fit_transform(X_processed_text)
                new_vect_s3= tfidf_s3.fit_transform(X_processed_text)

                # Melakukan prediksi dengan model RF
                result_a1 = model_a1.predict(new_vect_a1)  
                result_a2 = model_a2.predict(new_vect_a2)
                result_a3 = model_a3.predict(new_vect_a3)
                result_s1 = model_s1.predict(new_vect_s1)
                result_s2 = model_s2.predict(new_vect_s2) 
                result_s3 = model_s3.predict(new_vect_s3)  

                #output label gabungan hasil klasifikasi model aspek dan model sentimen
                atraksi_pred=[]
                a1=0
                for i in result_a1:
                    if i == 1 and result_s1[a1] == 1:
                        atraksi_pred.append("positif")
                    elif i == 1 and result_s1[a1] == 0 :
                        atraksi_pred.append("negatif")
                    else :
                        atraksi_pred.append("-")
                    a1 += 1

                amenitas_pred=[]
                a2=0
                for i in result_a2:
                    if i == 1 and result_s2[a2] == 1:
                        amenitas_pred.append("positif")
                    elif i == 1 and result_s2[a2] == 0 :
                        amenitas_pred.append("negatif")
                    else :
                        amenitas_pred.append("-")
                    a2 += 1

                aksesibilitas_pred=[]
                a3=0
                for i in result_a3:
                    if i == 1 and result_s3[a3] == 1:
                        aksesibilitas_pred.append("positif")
                    elif i == 1 and result_s3[a3] == 0 :
                        aksesibilitas_pred.append("negatif")
                    else :
                        aksesibilitas_pred.append("-")
                    a3 += 1

                # Create DataFrame baru
                df_temp = pd.DataFrame({'ulasan': X_processed_text,
                                    'atraksi': atraksi_pred,
                                    'amenitas': amenitas_pred,
                                    'aksesibilitas' : aksesibilitas_pred
                                    })

                st.markdown("<h2 style= color: white;'>Label Hasil Klasifikasi</h2>",unsafe_allow_html=True)
                st.write(df_temp)
            
                #Download Dataframe
                def convert_df(df_temp):
                    return df_temp.to_csv(index=False).encode('utf-8')
                
                csv = convert_df(df_temp)

                st.download_button(
                    "Download File",
                    csv,
                    "file.csv",
                    "text/csv",
                    key='download-csv'
                )

            if ml_choice == "Random Forest (Random Over Sampling)" :
                # Read data yang di upload
                data = pd.read_csv(uploaded_file)

                #Load model RF + ROS
                model_a1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a1_ROS.pickle','rb'))
                model_a2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a2_ROS.pickle','rb'))
                model_a3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a3_ROS.pickle','rb'))
                model_s1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s1_ROS.pickle','rb'))
                model_s2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s2_ROS.pickle','rb'))
                model_s3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s3_ROS.pickle','rb'))

                # Load kamus vocab RF + ROS dan membuat objek TfidfVectorizer dengan kamus vocab
                saved_vocabulary_a1 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a1_ROS.pickle", 'rb'))
                tfidf_a1 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a1)
                saved_vocabulary_a2 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a2_ROS.pickle", 'rb'))
                tfidf_a2 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a2)
                saved_vocabulary_a3 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_a3_ROS.pickle", 'rb'))
                tfidf_a3 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_a3)
                saved_vocabulary_s1 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s1_ROS.pickle", 'rb'))
                tfidf_s1 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s1)
                saved_vocabulary_s2 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s2_ROS.pickle", 'rb'))
                tfidf_s2 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s2)
                saved_vocabulary_s3 = pickle.load(open("D:\\skripsi\\streamlit\\model\\TFIDF RF ROS\\TFIDF_s3_ROS.pickle", 'rb'))  
                tfidf_s3 = TfidfVectorizer(sublinear_tf=True, vocabulary=saved_vocabulary_s3)
                
                #memuat ulasan dari data
                ulasan=data["ulasan"]

                #melakukan text preprocessing pada teks ulasan
                X_processed_text = [preprocessing_data(row) for row in ulasan]
                new_vect_a1= tfidf_a1.fit_transform(X_processed_text)
                new_vect_a2= tfidf_a2.fit_transform(X_processed_text)
                new_vect_a3= tfidf_a3.fit_transform(X_processed_text)
                new_vect_s1= tfidf_s1.fit_transform(X_processed_text)
                new_vect_s2= tfidf_s2.fit_transform(X_processed_text)
                new_vect_s3= tfidf_s3.fit_transform(X_processed_text)

                #Merubah data baru menajdi vektor tfidf
                result_a1 = model_a1.predict(new_vect_a1)  
                result_a2 = model_a2.predict(new_vect_a2)
                result_a3 = model_a3.predict(new_vect_a3)
                result_s1 = model_s1.predict(new_vect_s1)
                result_s2 = model_s2.predict(new_vect_s2) 
                result_s3 = model_s3.predict(new_vect_s3) 

                #output label gabungan hasil klasifikasi model aspek dan model sentimen
                atraksi_pred=[] # List untuk menyimpan prediksi atraksi
                a1=0 # Inisialisasi indeks
                for i in result_a1: # Iterasi melalui setiap elemen dalam list result_a1
                    if i == 1 and result_s1[a1] == 1:
                        atraksi_pred.append("positif")
                    elif i == 1 and result_s1[a1] == 0 :
                        atraksi_pred.append("negatif")
                    else :
                        atraksi_pred.append("-")
                    a1 += 1 # Perbarui indeks
                # Hasil prediksi atraksi tersimpan dalam list atraksi_pred
                
                amenitas_pred=[]
                a2=0
                for i in result_a2:
                    if i == 1 and result_s2[a2] == 1:
                        amenitas_pred.append("positif")
                    elif i == 1 and result_s2[a2] == 0 :
                        amenitas_pred.append("negatif")
                    else :
                        amenitas_pred.append("-")
                    a2 += 1

                aksesibilitas_pred=[]
                a3=0
                for i in result_a3:
                    if i == 1 and result_s3[a3] == 1:
                        aksesibilitas_pred.append("positif")
                    elif i == 1 and result_s3[a3] == 0 :
                        aksesibilitas_pred.append("negatif")
                    else :
                        aksesibilitas_pred.append("-")
                    a3 += 1
                
                #buat dataframe baru
                df_temp = pd.DataFrame({'ulasan': X_processed_text,
                                    'atraksi': atraksi_pred,
                                    'amenitas': amenitas_pred,
                                    'aksesibilitas' : aksesibilitas_pred
                                    })
                st.markdown("<h2 style= color: white;'>Label Hasil Klasifikasi</h2>",unsafe_allow_html=True)
                st.write(df_temp, width=100)
                
                #download dataframe
                def convert_df(df_temp):
                    return df_temp.to_csv(index=False).encode('utf-8')
                
                csv = convert_df(df_temp)

                st.download_button(
                    "Download File",
                    csv,
                    "file.csv",
                    "text/csv",
                    key='download-csv'
                )