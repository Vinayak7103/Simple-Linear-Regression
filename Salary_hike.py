#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns


# In[9]:


data1 = pd.read_csv("C:/Users/vinay/Downloads/Salary_Data.csv")
data1.head()


# In[18]:


model = smf.ols("data1['Salary'] ~ data1['YearsExperience']",data = data1).fit();model


# In[19]:


model.summary()


# In[20]:


Salary_hike = model.predict() ; Salary_hike


# In[28]:


salary_hike_predict =pd.DataFrame(Salary_hike,columns=['Salary_predict'])


# In[29]:


salary_hike_predict


# In[33]:


data1['salary_hike_predict'] = salary_hike_predict


# In[35]:


data1


# In[ ]:




