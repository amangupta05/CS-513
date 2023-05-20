# -*- coding: utf-8 -*-
"""CS 513 Group 5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qgUdmdQ1Q0ZUvnB9lBokWEJowrv_iioC

# Importing Libraries
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

"""# Loading Data"""

#importing the dataset
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/heart_disease_health_indicators_BRFSS2015.csv")

#seeing how our dataset looks like using first five rows
df.head()

#shape of the dataframe
df.shape

df.info()

df.describe()

#checking for the null values
df.isnull().sum()

#Replacing null values in numerical columns with their mean
for column in df:
    if df[column].dtype == 'float64':
        df[column].fillna(value = df[column].mean(), inplace = True)
df.isnull().sum()

"""# EDA"""

df.columns

catcol = ['', 'HighBP', 'HighChol', 'CholCheck',
       'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'DiffWalk', 'Sex', 'Education',
       'Income']

distcol = ['Age', 'MentHlth', 'PhysHlth', 'BMI']
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
sns.histplot(ax=axes[0,0], data=df, x=distcol[0])
sns.histplot(ax=axes[0,1], data=df, x=distcol[1])
sns.histplot(ax=axes[1,0], data=df, x=distcol[2])
sns.histplot(ax=axes[1,1], data=df, x=distcol[3])

plt.figure(figsize=(15,50))
for i,column in enumerate(catcol[1:]):
    plt.subplot(len(catcol), 2, i+1)
    plt.suptitle("Plot Value Count VS HeartAttack", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df, x=column, hue='HeartDiseaseorAttack')
    plt.title(f"{column}")
    plt.tight_layout()

bincol = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
         'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

plt.figure(figsize=(15,50))
for i,column in enumerate(bincol[1:]):
    plt.subplot(len(catcol), 2, i+1)
    plt.suptitle("Plot Value Count VS Education", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df, x=column, hue='Education')
    plt.title(f"{column}")
    plt.tight_layout()

plt.figure(figsize=(15,50))
for i,column in enumerate(bincol[1:]):
    plt.subplot(len(catcol), 2, i+1)
    plt.suptitle("Plot Value Count VS Income", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df, x=column, hue='Income')
    plt.title(f"{column}")
    plt.tight_layout()

import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
plt.figure(figsize=(12,10))
plt.title('Correlation Heatmap of Heart Disease')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
plt.show()

# Countplot on each feature
plt.figure(figsize=(20,60))
for i,column in enumerate(df.columns):
    plt.subplot(len(df.columns), 5, i+1)
    plt.suptitle("Plot Value Count", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df, x=column)
    plt.title(f"{column}")
    plt.tight_layout()

#data statistics
cor_matrix = df.corr().abs()
cor_matrix

#remove duplicate correlation diagonal
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
upper_tri

#drop both highly correlated columns
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7) or any(upper_tri[column] < 0.01)]
to_drop

#specify features and target columns
target = df['HeartDiseaseorAttack']
features = df.drop(to_drop, axis=1)
features = features.drop('HeartDiseaseorAttack',axis=1)
features.info()

#Data Scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

#data splitting
X_train, X_test, y_train, y_test = train_test_split(scaled_features,target,stratify=target, test_size=0.3)

"""#KNN

"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


#creating our knn classifier model with n = 3
knn_value_3 = KNeighborsClassifier(n_neighbors = 3)
knn_value_3.fit(X_train, np.ravel(y_train,order='C'))
accuracy_KNN=knn_value_3.score(X_test, y_test)
#Estimating the accuracy of knn classifier with 3 knn
print("Accuracy score ->", knn_value_3.score(X_test, y_test))

#creating confusion matrix
y_pred = knn_value_3.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""#Naive Bayes"""

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.naive_bayes import GaussianNB
Gnb = GaussianNB()
Gnb.fit(X_train, y_train)
y_pred = Gnb.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)
print("Naive Bayes Classifier Accuracy: ",accuracy_score(y_test, y_pred))
accuracy_NB=accuracy_score(y_test, y_pred)

"""#RandomForest"""

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Classifier Accuracy: ",accuracy_score(y_test, y_pred))
accuracy_RF=accuracy_score(y_test, y_pred)
print('Confusion matrix:-', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)



"""#DecisionTree"""

from sklearn.tree import DecisionTreeClassifier
#creating our Decision Tree Classifier - CART classifier is what is packaged into sklearn
clf_dt = DecisionTreeClassifier().fit(X_train, np.ravel(y_train,order='C'))
accuracy_DT=clf_dt.score(X_test, y_test)
#Estimating the accuracy
print("Accuracy score ->", clf_dt.score(X_test, y_test))

#creating confusion matrix
y_pred = clf_dt.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""# Adaboost"""

from sklearn.ensemble import AdaBoostClassifier
#creating a classifier using MLP
clf_ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.1).fit(X_train, np.ravel(y_train,order='C'))

accuracy_AD=clf_ada.score(X_test, y_test)
#Estimating the accuracy
print("Accuracy score ->", clf_ada.score(X_test, y_test))

#creating confusion matrix
y_pred = clf_ada.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""#ANN

"""

import tensorflow as tf
from tensorflow import keras

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
accuracy_ANN = model.evaluate(X_test, y_test)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
#creating confusion matrix
y_pred = model.predict(X_test)



# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""#SVM

"""

from sklearn import svm

model_SVM = svm.SVC(kernel='linear', C=1.0)

model_SVM.fit(X_train, y_train)

y_pred = model_SVM.predict(X_test)

accuracy_SVM = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy_SVM)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""#logistic regression

"""

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression object
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# Evaluate the accuracy of the model
accuracy_LR = lr.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy_LR * 100))
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# compute and print the classification report
report = classification_report(y_test, y_pred)
print(report)

"""Till now the accuracy is best for the ANN, after that Logistic regression and then for the SVM.


"""

import matplotlib.pyplot as plt
import numpy as np

# Creating a list of models and corresponding accuracy scores
models = ['KNN', 'Random Forest', 'ANN', 'SVM', 'Logistic Regression', 'AdaBoost', 'Decision Tree', 'Naive Bayes']
accuracy = [accuracy_KNN, accuracy_RF, accuracy_ANN, accuracy_SVM, accuracy_LR, accuracy_AD, accuracy_DT, accuracy_NB]

# Setting the width of each bar
bar_width = 0.5

# Setting the color for each bar
colors = ['#008fd5', '#fc4f30', '#e5ae37', '#6d904f', '#8b8b8b', '#c61aff', '#ffa600', '#20b2aa']


# Plotting the bar graph
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(models, accuracy, width=bar_width, color=colors)

# Setting the y-axis label
ax.set_ylabel('Accuracy Score', fontsize=14)

# Setting the x-axis label
ax.set_xlabel('Classification Models', fontsize=14)

# Setting the title of the plot
ax.set_title('Accuracy Scores of Classification Models', fontsize=18)

# Rotating the x-axis labels to 90 degrees for better readability
plt.xticks(rotation=90)

# Displaying the plot
plt.show()