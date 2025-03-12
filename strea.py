#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load dataset
st.title("NLP Sentiment Analysis & Classification App")
df=pd.read_csv("merch_sales.csv")
st.write("Dataset Overview:", df)
    
# Handling missing values
df.dropna(subset=['Review'], inplace=True)
    
# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

df['sentiment'] = df['vader_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

print(df)
st.write("Sentiment Analysis Results:")
st.write(df[['Review', 'sentiment_score', 'sentiment']])

    
# Text Classification
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Review'].astype(str))
y = (df['sentiment'] == 'Positive').astype(int)  # Binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:", classification_report(y_test, y_pred))


# In[ ]:


import pickle


# In[ ]:





# In[ ]:


model = pickle.load(open('NLP_1.pkl','rb'))
model


# In[ ]:





# In[ ]:





# In[ ]:




