# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:19:42 2026

@author: dhuma
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\dhuma\Downloads\emp-purches.csv")

x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train =sc.fit_transform(x_train)
x_test =sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(x_train,y_train)

y_pred =classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm 

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac


from sklearn.metrics import classification_report
cr= classification_report(y_test, y_pred)
cr


bias = classifier.score(x_train,y_train)
bias

variance =classifier.score(x_test,y_test)
variance


