# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:43:59 2026

@author: dhuma
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# Load dataset
dataset = pd.read_csv(r"C:\Users\dhuma\Downloads\employees_400.csv")

# Features (Age, Estimated Salary) and Target (Purchased)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=65, stratify=y)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# KNN Classifier
classifier_knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=3, algorithm='kd_tree')
classifier_knn.fit(x_train, y_train)

# Predictions
y_pred = classifier_knn.predict(x_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)


# Bias (train score) and Variance (test score)
bias = classifier_knn.score(x_train, y_train)
variance = classifier_knn.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)


##### FUTURE PREDICTION
dataset1 = pd.read_csv(r"C:\Users\dhuma\Downloads\empFU.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2, 3]].values
M = sc.transform(dataset1)   # use same scaler

d2['y_pred1'] = classifier_knn.predict(M)
d2.to_csv('final1.csv', index=False)

print("Saved file at:", os.path.join(os.getcwd(), 'final1.csv'))
