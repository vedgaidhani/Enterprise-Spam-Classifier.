import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
import joblib


st.set_page_config(page_title="ThreatDetect AI", page_icon="🛡️", layout="wide")


@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_ai_assets():

    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_ai_assets()


def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text



with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("ThreatDetect AI")
    st.caption("Production Instance v1.0")
    st.divider()
    st.success("⚡ Engine Status: Online & Optimized")
    st.markdown("""
    **Production Specs:**
    - Load Time: < 50ms
    - Database Dependency: Zero
    - Model: Multinomial Naive Bayes
    """)
    st.info("Engineer: Vedant Gaidhani")


st.title("🛡️ ThreatDetect AI: Live Message Analysis")
st.markdown("Instantly classify text messages using pre-trained NLP mathematical matrices.")
st.divider()

st.write("### 🚀 Test the Engine")
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Input suspected text message:", height=150, placeholder="Paste message here...")
    analyze_btn = st.button("Run Threat Analysis", type="primary", use_container_width=True)

with col2:
    st.write("#### Analysis Result")
    if analyze_btn:
        if user_input:
            with st.spinner("Analyzing NLP tokens..."):
                cleaned = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                
                if prediction == 'spam':
                    st.error("🚨 **THREAT DETECTED**")
                    st.write("This matches known malicious spam signatures.")
                else:
                    st.success("✅ **SAFE**")
                    st.write("No malicious signatures detected.")
        else:
            st.warning("Awaiting input...")