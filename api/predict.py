#!/usr/bin/env python
# coding: utf-8

# # Local predictions with FastAPI

# In[1]:


import requests
import pandas as pd


# ### Load data

# In[2]:


car = pd.read_csv("https://raw.githubusercontent.com/guber25/Car_acceptability/main/data/car.csv")


# ### Convert to json

# In[3]:


X = car.iloc[:, 1:-1]
json = [X.iloc[i].to_dict() for i in range(X.shape[0])]


# ### Predict

# In[4]:


url = 'http://127.0.0.1:8000/predict/'
res = requests.post(url, json=json)
y_pred = res.json()
print(y_pred)


