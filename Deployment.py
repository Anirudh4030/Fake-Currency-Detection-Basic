#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd



# In[9]:


# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Counterfeit Note Detection")
st.write("Enter the features of the banknote to check if it's real or fake.")

# Input fields
variance = st.number_input("Variance of Wavelet Transformed Image")
skewness = st.number_input("Skewness of Wavelet Transformed Image")
kurtosis = st.number_input("Kurtosis of Wavelet Transformed Image")
entropy = st.number_input("Entropy of the Image")

# Prediction
if st.button("Check Authenticity"):
    features = np.array([[variance, skewness, kurtosis, entropy]])
    prediction = model.predict(features)
    
    if prediction[0] == 0:
        st.success("The banknote is REAL!")
    else:
        st.error("The banknote is FAKE!")

# Instructions to run in Anaconda PowerShell
st.write("""
### How to Run the Deployment:
1. Open Anaconda PowerShell.
2. Navigate to the directory containing this script.
3. Run `streamlit run deployment.py`.
""")


# In[ ]:




