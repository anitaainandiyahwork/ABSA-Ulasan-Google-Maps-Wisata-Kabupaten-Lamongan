import base64
import streamlit as st
import re, string, ast
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# def set_background_image(image_path):
#     page_bg_img = '''
#         <style>
#         body {
#         background-image: url("''' + image_path + '''");
#         background-size: cover;
#         }
#         </style>
#         '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

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
        FILE_NORMALISASI_PATH = 'D:\\streamlit\\model\\normal.txt'

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
        FILE_STOPWORDS_PATH = 'D:\\streamlit\\model\\stopwords.txt'
        
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

    st.title('KLASIFIKASI TEKS ULASAN PENGUNJUNG OBJEK WISATA KABUPATEN LAMONGAN DI GOOGLE MAPS')
    with st.form(key='svmform'):
        kalimat = st.text_area("Masukkan Teks Ulasan Baru") 
        ml_choice = st.selectbox("Pilih Metode", ["Random Forest","Random Forest (Random Over Sampling)"])
        
        submit_button = st.form_submit_button(label='Analisis')

        if submit_button:
            if ml_choice == "Random Forest":

                model_a1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a1.pickle','rb'))
                model_a2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a2.pickle','rb'))
                model_a3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_a3.pickle','rb'))
                model_s1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s1.pickle','rb'))
                model_s2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s2.pickle','rb'))
                model_s3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF\\RFModel_s3.pickle','rb'))

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

                ulasan = kalimat
                X_processed_text = [preprocessing_data(ulasan)]

                new_vect_a1= tfidf_a1.fit_transform(X_processed_text)
                new_vect_a2= tfidf_a2.fit_transform(X_processed_text)
                new_vect_a3= tfidf_a3.fit_transform(X_processed_text)
                new_vect_s1= tfidf_s1.fit_transform(X_processed_text)
                new_vect_s2= tfidf_s2.fit_transform(X_processed_text)
                new_vect_s3= tfidf_s3.fit_transform(X_processed_text)

                result_a1 = model_a1.predict(new_vect_a1)  
                result_a2 = model_a2.predict(new_vect_a2)
                result_a3 = model_a3.predict(new_vect_a3)
                result_s1 = model_s1.predict(new_vect_s1)
                result_s2 = model_s2.predict(new_vect_s2) 
                result_s3 = model_s3.predict(new_vect_s3)  

                st.write("Hasil Pre-Processing")
                for x in X_processed_text:
                    st.header(x)

                col1, col2, col3= st.columns(3)
                #ATRAKSI
                col1.write("Atraksi")
                atraksi_pred=[]
                for i in result_a1:
                    if i == 1 and result_s1== 1:
                        atraksi_pred.append("positif")
                    elif i == 1 and result_s1 == 0 :
                        atraksi_pred.append("negatif")
                    else :
                        atraksi_pred.append("-")
                for i in atraksi_pred:
                    col1.header(i)
                
                #AMENITAS
                col2.write("Amenitas")
                amenitas_pred=[]
                for i in result_a2:
                    if i == 1 and result_s2 == 1:
                        amenitas_pred.append("positif")
                    elif i == 1 and result_s2 == 0 :
                        amenitas_pred.append("negatif")
                    else :
                        amenitas_pred.append("-")
                for k in amenitas_pred:
                    col2.header(k)

                #AKSESIBILITAS
                col3.write("Aksesibilitas")
                aksesibilitas_pred=[]
                for i in result_a3:
                    if i == 1 and result_s3 == 1:
                        aksesibilitas_pred.append("positif")
                    elif i == 1 and result_s3 == 0 :
                        aksesibilitas_pred.append("negatif")
                    else :
                        aksesibilitas_pred.append("-")
                for k in aksesibilitas_pred:
                    col3.header(k)

        if ml_choice == "Random Forest (Random Over Sampling)":
            model_a1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a1_ROS.pickle','rb'))
            model_a2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a2_ROS.pickle','rb'))
            model_a3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_a3_ROS.pickle','rb'))
            model_s1 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s1_ROS.pickle','rb'))
            model_s2 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s2_ROS.pickle','rb'))
            model_s3 = pickle.load(open('D:\\skripsi\\streamlit\\model\\MODEL RF ROS\\RFModel_s3_ROS.pickle','rb'))

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

            ulasan = kalimat
            X_processed_text = [preprocessing_data(ulasan)]

            new_vect_a1= tfidf_a1.fit_transform(X_processed_text)
            new_vect_a2= tfidf_a2.fit_transform(X_processed_text)
            new_vect_a3= tfidf_a3.fit_transform(X_processed_text)
            new_vect_s1= tfidf_s1.fit_transform(X_processed_text)
            new_vect_s2= tfidf_s2.fit_transform(X_processed_text)
            new_vect_s3= tfidf_s3.fit_transform(X_processed_text)

            result_a1 = model_a1.predict(new_vect_a1)  
            result_a2 = model_a2.predict(new_vect_a2)
            result_a3 = model_a3.predict(new_vect_a3)
            result_s1 = model_s1.predict(new_vect_s1)
            result_s2 = model_s2.predict(new_vect_s2) 
            result_s3 = model_s3.predict(new_vect_s3)  

            st.write("Hasil Pre-Processing")
            for x in X_processed_text:
                st.header(x)

            col1, col2, col3= st.columns(3)
            #ATRAKSI
            col1.write("Atraksi")
            atraksi_pred=[]
            for i in result_a1:
                if i == 1 and result_s1 == 1:
                    atraksi_pred.append("positif")
                elif i == 1 and result_s1 == 0 :
                    atraksi_pred.append("negatif")
                else :
                    atraksi_pred.append("-")
            for i in atraksi_pred:
                col1.header(i)
            
            #AMENITAS
            col2.write("Amenitas")
            amenitas_pred=[]
            for i in result_a2:
                if i == 1 and result_s2 == 1:
                    amenitas_pred.append("positif")
                elif i == 1 and result_s2 == 0 :
                    amenitas_pred.append("negatif")
                else :
                    amenitas_pred.append("-")
            for k in amenitas_pred:
                col2.header(k)

            #AKSESIBILITAS
            col3.write("Aksesibilitas")
            aksesibilitas_pred=[]
            for i in result_a3:
                if i == 1 and result_s3 == 1:
                    aksesibilitas_pred.append("positif")
                elif i == 1 and result_s3 == 0 :
                    aksesibilitas_pred.append("negatif")
                else :
                    aksesibilitas_pred.append("-")
            for k in aksesibilitas_pred:
                col3.header(k)