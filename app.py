import streamlit as st
import pandas as pd
import joblib
import ast

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Keyword Extractor", layout="wide")
st.title("استخراج الكلمات المفتاحية من الأبحاث العلمية 📄")

# --- 2. تحميل البيانات والموديل (باستخدام الكاش لزيادة السرعة) ---
@st.cache_data
def load_data():
    # قراءة الملف المضغوط مباشرة
    df = pd.read_csv('cleaned_papers.csv.zip', compression='zip')
    return df

@st.cache_resource
def load_model():
    # قراءة أداة TF-IDF
    return joblib.load('tfidf_model.pkl')

# تشغيل دوال التحميل
try:
    df = load_data()
    vectorizer = load_model()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الملفات: {e}")
    st.stop()

# --- 3. تصميم واجهة المستخدم (القائمة الجانبية) ---
st.sidebar.header("🔍 تصفح الأبحاث")
# عمل قائمة منسدلة بعناوين الأبحاث لاختيار واحد منها
selected_title = st.sidebar.selectbox("اختر عنوان البحث:", df['title'].tolist())

# استخراج بيانات البحث اللي المستخدم اختاره
selected_paper = df[df['title'] == selected_title].iloc[0]

# --- 4. عرض تفاصيل البحث ---
st.subheader("📌 العنوان (Title):")
st.info(selected_paper['title'])

st.subheader("📝 الملخص (Abstract):")
st.write(selected_paper['abstract'])

st.subheader("🔑 الكلمات المفتاحية (Top Keywords):")
# تحويل الكلمات المفتاحية من نص إلى قاموس (Dictionary) لعرضها بشكل منظم
try:
    keywords_dict = ast.literal_eval(selected_paper['keywords'])
    
    # عرض الكلمات في شكل أعمدة صغيرة
    cols = st.columns(len(keywords_dict))
    for col, (word, score) in zip(cols, keywords_dict.items()):
        with col:
            st.success(f"**{word}**\n\nScore: {score}")
except:
    st.write(selected_paper['keywords'])
