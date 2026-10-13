# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:03:31 2026

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
 


# training kernal_svm model on training set

from sklearn.svm import SVC

# Create classifier with RBF kernel
classifier = SVC(kernel='rbf', random_state=0)

# Fit the model
classifier.fit(x_train, y_train)


#predicting test set result
y_pred = classifier.predict(x_test)

#confusionmatrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)


# Bias (train score) and Variance (test score)
bias = classifier.score(x_train, y_train)
variance = classifier.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)




from sklearn.model_selection import cross_val_score

# K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f}".format(accuracies.std()))

# Visualization setup
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01), 
                     np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))








