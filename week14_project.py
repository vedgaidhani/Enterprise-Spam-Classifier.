import streamlit as st
import pandas as pd

st.title("Enterprise AI Spam Classifier")
st.subheader("Day 91: Data Pipeline & Exploration")

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["Label", "Message"]
    return df

try:
    data = load_data()
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Class Distribution (Spam vs. Safe)")
    distribution = data["Label"].value_counts()
    st.write(distribution)

    spam_pct = (distribution['spam'] / len(data)) * 100
    st.info(f"The dataset contains {spam_pct:.2f}% Spam and {100 - spam_pct:.2f}% Safe message.")

except FileNotFoundError:
    st.error("Please place the 'spam.csv' file inside your project folder to trigger the pipeline.")    