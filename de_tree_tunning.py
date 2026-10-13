# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:09:37 2026

@author: dhuma
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv(r"C:\Users\dhuma\Downloads\employees_400.csv")

# Features (Age, Estimated Salary) and Target (Purchased)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=50)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=5,       # limit depth
                                     min_samples_split=10,  # minimum samples to split
                                     min_samples_leaf=5,    # minimum samples per leaf
                                      random_state=50)
classifier.fit(x_train,y_train)

#Predicting the test test set results
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
