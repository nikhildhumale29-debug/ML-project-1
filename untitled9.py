# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:08:49 2026

@author: dhuma
"""


# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, 
        init='k-means++', 
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)   # inertia_ = WCSS
    
    
wcss = []
for i in range (1,11):
    kmeans=KMeans(
        n_clusters=i
        init=k_means++
        random_state42
        )    