# -*- coding: utf-8 -*-
"""
Logistic Regression (Classification)
Created on Thu Nov  4 07:42:17 2021
@author: Bagheri
"""
#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons     # many datasets
from sklearn.linear_model import LogisticRegression
from plot_2d_separator import plot_2d_separator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

X, y = make_moons(n_samples=200, noise=0.15, random_state=0)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='k', cmap=plt.cm.coolwarm)
plt.show()
X.shape
#X
#y

degree = 5
coeffs = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

plt.figure()

for C in coeffs:       # in sklearn: C inverse lambda! less C = more lambda = underfitting 
                                                     # more C = less lambda = overfitting , ...
                                                     # Fit: appropriate & less lambda!
    plt.subplot(2, 4, coeffs.index(C) + 1)
    plt.tight_layout()
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    log_reg = LogisticRegression(C=C)
    model = Pipeline([("poly_features", poly_features), ("logistic_regression", log_reg)])
    
    # train 
    model.fit(X, y)
    accuracy = model.score(X, y)
    
    # plot results
    title = "C = {:.2e} ({:.2f}%)"
    plot_2d_separator(model, X, fill=True)
    plt.scatter(X[:, 0], X[:, 1], s=15, c=y, alpha=0.5, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title.format(C, accuracy * 100), fontsize=10)

plt.show()