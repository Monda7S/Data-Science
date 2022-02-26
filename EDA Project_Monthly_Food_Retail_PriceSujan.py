#!/usr/bin/env python
# coding: utf-8

# # 1 . Load the Dataset

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 2 . Cleaning the Dataset

# In[31]:


df = pd.read_csv("Monthly_Food_Retail_Price.csv")


# In[32]:


df['Category'].unique()


# In[33]:


df['Unit'].unique()


# In[34]:


df = df.drop(['Category', 'Unit'],axis = 1)


# In[35]:


df = df.rename(columns = {'Retail Price': 'Retails_Price'}, inplace = False)
df


# # The Shape Data

# In[36]:


df.shape


# In[37]:


df.ndim


# In[38]:


df.size


# # Print all columns Name

# In[39]:


df.columns


# In[40]:


df.index


# # Find the data type of all the columns

# In[15]:


df.dtypes


# # Print information & Describe

# In[16]:


df.info()


# In[17]:


df.describe()


# # 3 . The missing values (Column_wise)

# In[18]:


df.isnull().sum()


# In[41]:


miss_df = df[df["Commodity"].isnull()]
miss_df


# # The missing Values (Column_Wise)

# In[20]:


df.isnull().sum(axis=0).sort_values(ascending=False)


# # The missing values (row_wise)
# 

# In[21]:


df.isnull().sum(axis=1).sort_values(ascending=False)


# # Column having at least one missing value

# In[22]:


df.isnull().any()


# In[23]:


df.index


# # Column Having all missing values

# In[24]:


df.isnull().all(axis=0)


# # Rows having all missing values

# In[42]:


df.isnull().all(axis=1).sum()


# # 4 . Save the clean data in CSV and JSON format

# In[43]:


import csv


# In[44]:


df.to_csv("Sujan_Clean_EDA_Data.csv")


# In[45]:


df.to_json("Sujan_Clean_EDA_Data.json")


# In[46]:


df.shape


# # EDA with Visualization if required

# # 5 . Load Cleaned Dataset

# In[47]:


df2 = df
df2


# # 6 . Which date has highest price of wheat

# In[48]:


com = df.loc[(df['Commodity'] == 'Wheat')]
com


# In[59]:


X = df2.iloc[:,2:4].values
y = df2.iloc[:,-1]


# In[57]:


a = y.value_counts()


# # Which date has highest price of wheat?

# In[149]:


# max_price= com['Retails_Price'].max()
# max_price
com=df.loc[(df['Commodity'] =='Wheat')]
com
max_price=com.loc[com['Retails_Price'].idxmax()]
max_price


# # 7 Which is the highest price of wheat state wise

# In[147]:


state_wise=com.groupby('State').max().reset_index()
state_wise

# state wise = com.groupby('State')['Retails_Price'].max()
# state_wise
# state_wise_sort=state_wise.reset_index().sort_values(['Retails_Price'], ascending = False)
# state_wise_sort.reset_index().drop(['index'],axis=1)


# # Split into features and Target

# In[89]:


X = df.iloc[:,2:4]
y = df.iloc[:,-1]


# In[90]:


a = y.value_counts()


# In[92]:


plt.pie(a,autopct="%0.2f")
plt.show()


# # Features scaling
# 

# In[104]:


def cofficient(x,y):
    
    s_x = sum(x)
    s_y = sum(y)
    s_x2 = sum(x**2)
    s_xy = sum(x*y)
    n = len(x)
    
    b0 = (s_y*s_x2 - s_x*s_xy)/(n*s_x2 - (s_x)**2)
    b1 = (n*s_xy - s_x*s_y)/(n*s_x2 - (s_x)**2)
    
    return b0,b1


# In[111]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# # If the Commodity is mustard oil, how many varietes are available in mustard oil

# In[123]:


df2=df.loc[df['Commodity'] == 'Mustard oil','Variety'].iloc(0)
print(df2)


# In[ ]:




