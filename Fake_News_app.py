# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:14:06 2025

@author: Gireesh
"""

#importing the libraries
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model

# Download NLTK assets
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load model and vectorizer
model = load_model('C:/Users/HP/OneDrive/Documents/projects/Fake News prediction system/FakeNewsNet.keras')
vectorizer = joblib.load('C:/Users/HP/OneDrive/Documents/projects/Fake News prediction system/vectorizer.joblib')

# App styling
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff6347;'>🕵️‍♂️ Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news headline and its domain to predict if it's fake or real.</p>", unsafe_allow_html=True)

# Input section
with st.form("news_form"):
    input_title = st.text_area("🔤 News Title", "")
    input_domain = st.text_input("🌐 Source Domain (e.g. www.abcnews.com)", "")
    submitted = st.form_submit_button("🚀 Predict")

# Preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# Prediction logic
if submitted:
    if input_title.strip() == "" or input_domain.strip() == "":
        st.warning("⚠️ Please fill in both fields.")
    else:
        user_input = preprocess(input_title + " " + input_domain)
        features = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(features)

        st.markdown("---")
        if float(prediction[0]) <= 0.5:
            st.error("🧨 This is likely **Fake News**!", icon="🚫")
        else:
            st.success("✅ This appears to be **Real News**.", icon="📰")

        confidence = float(prediction[0][0])
        confidence = confidence if confidence >= 0.5 else 1 - confidence
        st.markdown("🧠 *Model confidence: {:.2f}%*".format(confidence * 100))
