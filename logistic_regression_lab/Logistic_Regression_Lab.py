#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  24 03:05:54 2019

@author: zemmouri
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *

import random



##### Example with randomly generated data 

X1 = np.random.randn(50, 2) + 2.0
y1 = np.zeros(50)
X2 = np.random.randn(50, 2) + 4.0
y2 = np.ones(50)

X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

ind = list(range(X.shape[0]))
random.shuffle(ind)

X = X[ind, :]
y = y[ind]

print('Shape X = ', X.shape)
print('Shape y = ', y.shape)

## Plot data

mask1 = (y == 1)
mask0 = (y == 0)
X1 = X [mask1, :]
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='class 1')
X0 = X [mask0, :]
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='class 0')
plt.legend(loc='upper right')
plt.show()

## Run Logistic Regression

iterations = 2000
model = LogisticRegressionClassifier(learning_rate=0.1, 
                                     max_iterations = iterations)

model.fit(X, y)

## Plot Cost curve

costs = model.costs
iterations = np.arange(0, iterations, 100)

plt.plot(iterations, model.costs)
plt.title('Convergence of G. D.')
plt.xlabel('Number Iterations')
plt.ylabel('Cost J')
plt.show()


## Plot data with Decision Boundary

w, b = model.w, model.b

xd = np.array([np.min(X[:, 0]) - 1 , np.max(X[:, 0]) + 1])
yd = - (xd * w[0, 0] + b) / w[1, 0]

mask1 = (y == 1)
mask0 = (y == 0)
X1 = X [mask1, :]
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='class 1')
X0 = X [mask0, :]
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='class 0')

plt.plot(xd, yd, color='g')

plt.legend(loc='upper right')
plt.show()

model.score(X, y)





#### Example with a file dataset

## Load data

data = np.loadtxt('dataset1.txt', delimiter=',')
print(data.shape)
X = data[:, (0, 1)]
y = data[:, 2].astype(int)
print(X.shape)
print(y.shape)


# Plot data
mask1 = (y == 1)
mask0 = (y == 0)
X1 = X [mask1, :]
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='Admis')
X0 = X [mask0, :]
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='Non Admis')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.legend(loc='upper right')
plt.show()


## Run Logistic Regression

iterations = 5000

model = LogisticRegressionClassifier(learning_rate=0.001, 
                                   max_iterations = iterations)

model.fit(X, y)


## Plot Cost curve

costs = model.costs
iterations = np.arange(0, iterations, 100)

plt.plot(iterations, model.costs)
plt.title('Convergence of G. D.')
plt.xlabel('Number Iterations')
plt.ylabel('Cost J')
plt.show()


## Plot data with Decision Boundary

w, b = model.w, model.b

xd = np.array([np.min(X[:, 0]) - 1 , np.max(X[:, 0]) + 1])
yd = - (xd * w[0, 0] + b) / w[1, 0]

mask1 = (y == 1)
mask0 = (y == 0)
X1 = X [mask1, :]
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='class 1')
X0 = X [mask0, :]
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='class 0')

plt.plot(xd, yd, color='g')

plt.legend(loc='upper right')
plt.show()

model.score(X, y)






#### Logistic Regression using sklearn

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='saga', multi_class='ovr').fit(X, y)
accuracy = clf.score(X, y)
print('Accuracy on training dataset : ', accuracy)

clf = LogisticRegression(solver='lbfgs', multi_class='ovr').fit(X, y)
accuracy = clf.score(X, y)
print('Accuracy on training dataset : ', accuracy)
