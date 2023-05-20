#!/usr/bin/env python
# coding: utf-8

# # HW_04_NB
# #Name-Aman Gupta #cwid-20018346

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


df = pd.read_csv("breast-cancer-wisconsin .csv",na_values = "?")


# In[16]:


df.head()


# In[17]:


df.info()


# In[18]:


df.columns = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
              'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
              'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis']

df['diagnosis'] = df['diagnosis'].map({2: 'benign', 4: 'malignant'})

df = df.replace('?', np.nan)
df = df.dropna()

cols = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
        'bland_chromatin', 'normal_nucleoli', 'mitoses']
df[cols] = df[cols].astype('category')


# In[19]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[20]:


X= df.iloc[:,:-1].values
y= df.iloc[:,10].values


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


# In[22]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)


# In[23]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

acc = accuracy_score(y_test, y_pred)
print("Accuracy : {:.2f}%".format(acc * 100))


# In[25]:


print(confusion_matrix(y_test, y_pred))


# In[26]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




