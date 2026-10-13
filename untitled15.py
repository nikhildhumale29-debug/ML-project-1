# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:25:28 2026

@author: dhuma
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\dhuma\Downloads\employees_400.csv")
x= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#FEACHER SCALING 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)

#splitting the dataset into the trainning set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=50)



from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimators=100,       # number of trees
                              learning_rate=0.1,      # step size shrinkage
                              max_depth=3,            # depth of trees
                              random_state=0,
                              use_label_encoder=False,
                              eval_metric='logloss'  )

classifier.fit(x_train, y_train)

# Predicting test set results
y_pred = classifier.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Cross Validation
# -------------------------------
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("CV Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("CV Std Dev: {:.2f}".format(accuracies.std()))

# -------------------------------
# Bias–Variance Analysis
# -------------------------------
bias = classifier.score(x_train, y_train)
variance = classifier.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)
