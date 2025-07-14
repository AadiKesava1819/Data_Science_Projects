#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model = pickle.load(open('model1.pkl','rb'))


# In[3]:


st.title("Sentiment Analyzer")


# In[4]:


user_input = st.text_area("Enter your review here:", height=150)


# In[5]:


import re
import spacy
nlp= spacy.load('en_core_web_sm')
def data_clean(x):
  data = ' '.join(re.findall('\w+',x))
  data1 = nlp(data)
  cleaned_text = [token.lemma_ for token in data1 if not token.is_stop and not token.is_punct and not token.is_bracket and not token.is_digit and not token.is_currency]
  return ' '.join(cleaned_text)


# In[6]:


vectorizer = pickle.load(open('model2.pkl','rb'))


# In[20]:


if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = data_clean(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            st.markdown("### Prediction Result")
            if prediction == 2:
                st.markdown('Positive', unsafe_allow_html=True)
            elif prediction == 1:
                st.markdown('Negetive', unsafe_allow_html=True)
            else:
                st.markdown('Neutral', unsafe_allow_html=True)
    


# In[ ]:




