# 🛡️ ThreatDetect AI: Enterprise Spam Classification Engine

**Live Application:** [View on Streamlit Cloud](https://vedant-threatdetect.streamlit.app/)

## 📖 Overview
ThreatDetect AI is a decoupled Natural Language Processing (NLP) application designed to instantly classify text messages as safe or malicious. Built with an emphasis on production-ready architecture, this project separates the heavy machine learning training pipeline from the lightweight, lightning-fast user interface.

## ⚙️ Tech Stack & Architecture
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Multinomial Naive Bayes)
* **NLP Processing:** NLTK (Stopwords, Text Normalization), TF-IDF Vectorization
* **Frontend UI:** Streamlit
* **Deployment & Hosting:** Streamlit Community Cloud
* **Serialization:** Joblib (Binary model export)

### The Decoupled Architecture
To achieve `< 50ms` inference times in the cloud, the architecture was split:
1. **The Backend (`week14_project.py`):** Ingests the 5,000+ row dataset, runs the NLP cleaning pipeline, trains the Naive Bayes model, and serializes the "brain" into binary `.pkl` files. 
2. **The Frontend (`app.py`):** A lightweight cloud UI that has zero database dependencies. It only loads the pre-trained `.pkl` binaries to run instant text evaluation for the end user.

## 📊 Model Performance
The model was optimized for **Precision** to completely eliminate False Positives (preventing safe messages from being hidden).
* **Baseline Accuracy:** 96.59%
* **False Positives:** 0 (Perfect safety rate for legitimate messages)
* **False Negatives:** 38
* **True Spam Caught:** 112

## 🚧 Challenges Faced & Engineering Solutions
Building this pipeline presented several real-world engineering hurdles that had to be solved:

1. **The Re-run Bottleneck:** Initially, Streamlit re-ran the entire 5,000-message training pipeline every time a user clicked a button. **Solution:** Architected a decoupled system using `joblib` to freeze-dry the trained model, dropping load times from seconds to milliseconds.
2. **Cloud Dependency Failures:** During deployment, the cloud server threw `[Errno 11001]` network errors because it lacked the English dictionary resources. **Solution:** Implemented a targeted `requirements.txt` environment file and cached the `nltk.download('stopwords')` command to safely ping external servers during the cloud build phase.
3. **Training Data Bias:** While testing live inference, the AI easily caught 2012-era "Prize Jackpot" scams but failed to catch modern 2026 "Bank Account Locked" phishing texts. **Learning:** An AI is only as smart as its dataset. Naive Bayes is inherently conservative; to catch modern threats, the model requires an updated dataset with contemporary social engineering vocabulary.

## 🚀 Quick Start (Run Locally)
If you wish to run this engine on your own machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/vedgaidhani/Enterprise-Spam-Classifier.](https://github.com/vedgaidhani/Enterprise-Spam-Classifier.)

   