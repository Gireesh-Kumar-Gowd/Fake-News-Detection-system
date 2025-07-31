# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:45:03 2025

@author: Gireesh
"""

import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from transformers import pipeline

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff6347;'>🕵️‍♂️ Fake News Detection System</h1>", unsafe_allow_html=True)

# Preprocessing for Keras model
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

@st.cache_resource
def load_keras_model():
    return load_model('C:/Users/HP/OneDrive/Documents/projects/Fake News prediction system/FakeNewsNet.keras')

@st.cache_resource
def load_vectorizer():
    return joblib.load('C:/Users/HP/OneDrive/Documents/projects/Fake News prediction system/vectorizer.joblib')

@st.cache_resource
def load_transformers_pipeline():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

keras_model = load_keras_model()
vectorizer = load_vectorizer()
transformer_pipeline = load_transformers_pipeline()

# UI
st.markdown("Choose the model backend:")
model_choice = st.selectbox("🔧 Select Model", ["Keras Model (Local)", "Hugging Face Transformers (Online)"])

# Input fields
with st.form("news_form"):
    input_title = st.text_area("🔤 News Title", value=st.session_state.get("input_title", ""))
    input_domain = st.text_input("🌐 Source Domain (e.g. www.abcnews.com)", value=st.session_state.get("input_domain", ""))
    submitted = st.form_submit_button("🚀 Predict")

# Prediction
if submitted:
    if input_title.strip() == "" or input_domain.strip() == "":
        st.warning("⚠️ Please fill in both fields.")
    else:
        with st.spinner("🔍 Analyzing..."):
            full_text = input_title + " " + input_domain

            if model_choice == "Keras Model (Local)":
                user_input = preprocess(full_text)
                features = vectorizer.transform([user_input]).toarray()
                prediction = keras_model.predict(features)
                confidence = float(prediction[0][0])
                label = "Real News" if confidence > 0.5 else "Fake News"
                confidence = confidence if confidence > 0.5 else 1 - confidence

            else:  # Hugging Face Transformers
                result = transformer_pipeline(full_text)[0]
                label = "Real News" if "LABEL_1" in result["label"] else "Fake News"
                confidence = result["score"]

        st.markdown("---")
        if label == "Fake News":
            st.error(f"🧨 Likely **Fake News**", icon="🚫")
        else:
            st.success(f"✅ Appears to be **Real News**", icon="📰")
        st.markdown(f"🧠 *Model confidence: `{confidence * 100:.2f}%`*")
        st.markdown(f"🧾 *Prediction source: **{model_choice}***")
