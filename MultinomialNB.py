# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:04:50 2026

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
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=65)


# Feature scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.naive_bayes import MultinomialNB
classifier_nb =  MultinomialNB()
classifier_nb.fit(x_train, y_train)


y_pred = classifier_nb.predict(x_test)


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
bias = classifier_nb.score(x_train, y_train)
variance = classifier_nb.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)

%pip install lazypredict

from lazypredict.Supervised import LazyClassifier

# Create LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True)

# Train and evaluate all classifiers
models, predictions = clf.fit(
    x_train,
    x_test,
    y_train,
    y_test
)

print(models)





