import joblib 
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#----------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added at day 94


@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()

st.set_page_config(page_title="ThreatDetect AI",page_icon="🛡", layout="wide")
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("ThreatDetect AI")
    st.caption("Enterprise NLP Spam Filter")
    st.divider()
    st.markdown("""
    **Engine Specs**
    - Algoritham: Multinomial Naive Bayes
    - Vectorization: TF-IDK Logarithms
    - Preprocessing: NLTK Stopwords
    """)
    st.info("Built by Vedant Gaidhani | AI Application Engineer")

st.title("🛡 ThreatDetect AI: Live Message Analysis")
st.markdown("Instantly classify text messages using Natural Language Processing.")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["Label", "Message"]
    return df

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

try:
    data = load_data()


    with st.expander("View Day 91 & 92 Data Pipeline Steps"):
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        st.write("### Class Distribution")
        distribution = data["Label"].value_counts()
        st.write(distribution)

        st.write("### Data Cleaning")
        with st.spinner("Running text through NLP pipeline..."):
            data["Cleaned_Message"] = data["Message"].apply(clean_text)
        st.dataframe(data[['Message','Cleaned_Message']].head(5))
        

    Vectorizer =  TfidfVectorizer()
    X = Vectorizer.fit_transform(data['Cleaned_Message'])
    y = data['Label']

    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    model = MultinomialNB()
    model.fit(X_train, y_train)

    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(Vectorizer, 'tfidf_vectorizer.pkl')
    st.toast("🧠 AI Brain saved to disk!")


    st.write("### Day 94: AI Evaluation & Business Logic")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)


    st.write("#### The Confusion Matrix Breakdown")
    st.write(f"- ✅ **True Safe(Ham):** '{cm[0][0]}' (Safe message correctly identified)")
    st.write(f"- 🚨 **FALSE POSITIVE:** '{cm[0][1]}' (safe message WRONGLY flagged as spam. *This is dangerous!*)")
    st.write(f"- ⚠ **FALSE NEGATIVE:** '{cm[1][0]}' (spam messages that slipped through into the inbox)")
    st.write(f"- 🎯 **True Spam:** '{cm[1][1]}' (Spam messages correctly caught and blocked)")

    st.write("###  Detailed Classification Report") 
    st.code(classification_report(y_test, y_pred))

#----------day 95...

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

                with st.spinner("Analyzing the NLP tokens.."):
                    cleand_input = clean_text(user_input)
                    Vectorized_input = Vectorizer.transform([cleand_input])
                    prediction = model.predict(Vectorized_input)[0]

                    if prediction == "spam":
                        st.error("🚨 **ALERT: THREAT DETECTED** ")
                        st.write("This matches known spam signatures.")
                    else:
                        st.success("✅ **SAFE:** ")
                        st.write("No malicioous signatures detected.")
            else:
                st.warning("Awaiting inputs...")

   
except FileNotFoundError:
    st.error("Please place the 'spam.csv' file inside your projected folder to trigger the pipeline.")