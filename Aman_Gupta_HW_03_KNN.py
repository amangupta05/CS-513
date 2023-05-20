#!/usr/bin/env python
# coding: utf-8

# # HW_03_knn
# #Name-Aman Gupta #cwid-20018346

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("breast-cancer-wisconsin .csv",na_values = "?")


# In[3]:


df.head()


# In[ ]:





# In[4]:


df.info()


# In[5]:


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


# In[6]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[7]:


X= df.iloc[:,:-1].values
y= df.iloc[:,10].values


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
classifier_3 = KNeighborsClassifier(n_neighbors=3)

classifier_3.fit(X_train, y_train)

y_pred_3=classifier_3.predict(X_test)


# In[11]:


classifier_5 = KNeighborsClassifier(n_neighbors=5)
classifier_5.fit(X_train, y_train)
y_pred_5=classifier_5.predict(X_test)


# In[12]:


classifier_10 = KNeighborsClassifier(n_neighbors=10)
classifier_10.fit(X_train, y_train)

y_pred_10=classifier_10.predict(X_test)


# In[94]:





# In[13]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



acc_3 = accuracy_score(y_test, y_pred_3)
print("Accuracy with k=3: {:.2f}%".format(acc_3 * 100))

acc_5 = accuracy_score(y_test, y_pred_5)
print("Accuracy with k=5: {:.2f}%".format(acc_5 * 100))

acc_10 = accuracy_score(y_test, y_pred_10)
print("Accuracy with k=10: {:.2f}%".format(acc_10 * 100))


# In[14]:


print(confusion_matrix(y_test, y_pred_3))
print(confusion_matrix(y_test, y_pred_5))
print(confusion_matrix(y_test, y_pred_10))


# In[15]:


print(classification_report(y_test, y_pred_3))
print(classification_report(y_test, y_pred_5))
print(classification_report(y_test, y_pred_10))

