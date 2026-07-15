import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#----------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()

st.title("Enterprice AI Spam Classifier")
st.subheader("Day 92: Feature Extraction & Data plitting")

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

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Class Distribution (Spam vs. Safe)")
    distribution = data["Label"].value_counts()
    st.write(distribution)

    spam_pct = (distribution['spam'] / len(data)) * 100
    st.info(f"The dataset contains {spam_pct:.2f}%Spam and {100 - spam_pct:.2f}% Safe messages.")


    st.write("### Data Cleaning (NLP in Action)")
    with st.spinner("Running text through NLP cleaning pipeline..."):
        data['Cleaned_Message'] = data['Message'].apply(clean_text)

    st.dataframe(data[['Message', 'Cleaned_Message']].head(10))

    st.write("### Day 92: Converting Words to Math (TF-IDF)")

    Vectorizer = TfidfVectorizer()

    with st.spinner("Calculating TF-IDF Logarithms..."):
        X = Vectorizer.fit_transform(data["Cleaned_Message"])
        y = data['Label']


    st.write("### Training & Testing Split")
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.success(f"Data Succssfully Split! 80% Training Data: {X_train.shape[0]} messages | 20% Testing Data: {X_test.shape[0]} messages. ")

    st.write("### Day 93: Training the AI (Naive Bayes)")
    model = MultinomialNB()

    with st.spinner("Training the Naive Bayes model..."):  # spinner = loading animation...
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy =  accuracy_score(y_test, predictions)

    st.success(f"Model Training Complete! Baseline Accuracy: {accuracy * 100:.2f}%")

except FileExistsError:
    st.error("Please place the 'spam.csv' file inside your projected folder to trigger the pipeline.")