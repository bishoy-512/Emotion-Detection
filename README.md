# Emotion Detection App 😃

A simple NLP project that classifies English text into three emotions: **Joy**, **Fear**, and **Anger**.  
This project uses **TF-IDF / CountVectorizer** with **RandomForest** or **Naive Bayes** for classification.

---

## Dataset

The dataset used is from [Kaggle: Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) and contains two columns:

- `Comment`: Text about a situation or event.  
- `Emotion`: Emotion label (`joy`, `fear`, `anger`).  

This is a **multi-class classification** problem with 3 classes.

---

## Features

- Preprocessing with **spaCy** (removing stop words, punctuation, and lemmatization).  
- Trained a **RandomForest pipeline** with **TF-IDF** features.  
- Deployed using **Streamlit** for interactive testing.  

---
