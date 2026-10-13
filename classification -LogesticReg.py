# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:02:49 2026

@author: dhuma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Load dataset
dataset = pd.read_csv(r"C:\Users\dhuma\Downloads\employees_400.csv")

# Features (Age, Estimated Salary) and Target (Purchased)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Train-test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predictions
y_pred = classifier.predict(x_test)

# Confusion Matrix
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
bias = classifier.score(x_train, y_train)
variance = classifier.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)



dataset1 = pd.read_csv(r"C:\Users\dhuma\Downloads\empFU.csv")

d2 =dataset1.copy()

dataset1 =dataset1.iloc[:,[2,3]].values 

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 =pd.DataFrame()
y_pred1
d2 ['y_pred1'] =classifier.predict(M)
d2.to_csv('final1.csv')

#to get path

import os
os.getcwd()


