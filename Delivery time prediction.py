#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
data = pd.read_csv("delivery_time.csv")
data.head(25)


# In[3]:


model = smf.ols("data['Delivery Time'] ~ data['Sorting Time']",data = data).fit();model


# In[4]:


sns.regplot(x="Sorting Time", y="Delivery Time", data=data)


# In[5]:


model.summary()


# In[6]:


(model.rsquared,model.rsquared_adj)


# In[7]:


prediction = model.predict()


# In[8]:


Delivery_time_predict =pd.DataFrame(prediction,columns=['Delivery Time'])


# In[9]:


Delivery_time_predict


# In[11]:


data['Delivery_time_predict'] = Delivery_time_predict


# In[12]:


data


# In[ ]:




