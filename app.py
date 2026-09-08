import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import docx

# --- 1. تحميل مكتبات اللغة ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_data()

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="Keyword Extractor App", layout="centered")
st.title("استخراج الكلمات المفتاحية 🚀")
st.write("تقدر تلصق النص، أو ترفع أي ملف (TXT, CSV, PDF, Word) عشان نستخرج الكلمات المفتاحية منه.")

# --- 3. تحميل الموديل ---
@st.cache_resource
def load_model():
    return joblib.load('tfidf_model.pkl')

try:
    vectorizer = load_model()
except Exception as e:
    st.error("مش قادر ألاقي ملف الموديل 'tfidf_model.pkl'.")
    st.stop()

# --- 4. دالة التنظيف ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) >= 3]
    return ' '.join(words)

# --- 5. دالة الاستخراج والعرض ---
def process_and_show_keywords(input_text):
    with st.spinner("جاري تحليل النص واستخراج الكلمات..."):
        cleaned_input = clean_text(input_text)
        tfidf_matrix = vectorizer.transform([cleaned_input])
        feature_names = vectorizer.get_feature_names_out()
        row_scores = tfidf_matrix.toarray()[0]
        
        top_indices = np.argsort(row_scores)[-5:][::-1]
        
        st.subheader("🔑 أهم الكلمات المفتاحية:")
        found_keywords = False
        cols = st.columns(5)
        
        for col, idx in zip(cols, top_indices):
            score = row_scores[idx]
            if score > 0:
                found_keywords = True
                with col:
                    st.success(f"**{feature_names[idx]}**\n\n{score:.3f}")
        
        if not found_keywords:
            st.info("لم يتم العثور على كلمات مفتاحية قوية. جرب إدخال نص علمي أطول.")

# --- 6. واجهة المستخدم (التبويبات) ---
tab1, tab2 = st.tabs(["📝 إدخال نص يدوياً", "📁 رفع ملف"])

# التبويب الأول: إدخال النص
with tab1:
    user_input = st.text_area("أدخل النص أو الملخص (Abstract) هنا:", height=200)
    if st.button("استخراج الكلمات 🔍", key="btn_text"):
        if user_input.strip() == "":
            st.warning("الرجاء إدخال نص أولاً!")
        else:
            process_and_show_keywords(user_input)

# التبويب الثاني: رفع الملف
with tab2:
    # ضفنا دعم لـ pdf و docx
    uploaded_file = st.file_uploader("ارفع ملف (TXT, CSV, PDF, DOCX)", type=['txt', 'csv', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        file_text = ""
        
        # 1. قراءة ملفات Text
        if file_name.endswith('.txt'):
            file_text = uploaded_file.read().decode('utf-8')
            
        # 2. قراءة ملفات PDF
        elif file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    file_text += extracted + "\n"
                    
        # 3. قراءة ملفات Word
        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            file_text = "\n".join([para.text for para in doc.paragraphs])
            
        # 4. قراءة ملفات CSV
        elif file_name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
            st.success("✅ تم قراءة ملف البيانات بنجاح.")
            col_to_process = st.selectbox("اختر العمود اللي بيحتوي على النص:", df_uploaded.columns)
            row_index = st.number_input("اختر رقم الصف (البحث) اللي عايز تفحصه:", min_value=0, max_value=len(df_uploaded)-1, step=1)
            file_text = str(df_uploaded[col_to_process].iloc[row_index])

        # عرض النص المستخرج وزرار التحليل (للملفات غير الـ CSV)
        if file_text.strip() != "":
            st.info("✅ تم قراءة الملف بنجاح.")
            st.write("**النص المستخرج (أول 300 حرف):**", file_text[:300] + "...")
            
            if st.button("استخراج الكلمات من الملف 🔍", key="btn_file"):
                process_and_show_keywords(file_text)
        elif not file_name.endswith('.csv'):
            st.warning("الملف يبدو فارغاً أو لم أتمكن من قراءة النص منه.")
