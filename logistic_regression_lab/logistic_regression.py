#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  24 02:11:06 2019

@author: zemmouri
"""


import numpy as np


class LogisticRegressionClassifier :
    def __init__(self, learning_rate=0.1, regularization=0.0, max_iterations=1500):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iterations = max_iterations

        self.w = np.array([0.0])
        self.b = 0.0

        self.x = np.array([0.0])
        self.y = np.array([0.0])

        self.costs = []

    

    def sigmoid (self, x):
        """
        Function that computes the sigmoid of x.
        Arguments :
            x : a scalar or numpy array of any size.
        Return :
            scalar or numpy array sigmoid of x
        """
        return 1.0 / (1.0 + np.exp(-x))
    
    def init_param (self, num_features):
        """
        Function that initializes parameters :
        w to a vector of zeros of shape (num_features, 1)
        b to 0.
        
        """
        self.w = np.zeros((num_features, 1)) * 0.0
        self.b = 0.0
    

    def cost_grad (self):
        """
        Function that computes cost and its gradient for logistic regression with regularization.
        Arguments :
            
        Return :
            J : log-likelihood cost for logistic regression with regularization
            dw : gradient of J with respect to w, dw has the same shape as w
            db : gradient of J with respect to b, db has the same shape as b 
        """
        m = self.x.shape[0]

        yhat = self.sigmoid(np.dot(self.x, self.w) + self.b)

        J = -( np.dot(self.y.T, np.log(yhat)) + np.dot((1.0 - self.y).T , np.log(1.0 - yhat)) ) / m
        
        J = J + self.regularization * np.sum(self.w ** 2) / (2*m)

        J = J.item()

        dw = (np.dot(self.x.T, yhat - self.y) + self.regularization * self.w) / m
        db = np.sum(yhat - self.y) / m

        return J, dw, db
    

    def optimize (self):
        """
        Function that optimizes w and b using Gradient Descent algorithm

        Returns:
            costs : list of all the costs computed during the optimization, to be used to plot the learning curve.
        """
        print('\nRunning Gradient Descent : \n--------------------------------\n')
        self.costs = []
        for i in range(self.max_iterations):
            J, dw, db = self.cost_grad()
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if i%100 == 0: 
                self.costs.append(J)
                print ("Cost after iteration %i = %f" %(i, J))
        
        

    def fit (self, X, y):
        """
        Function that fits the model according to the given training data.
        that is optimize model parameters w and b according to X and y.
        """
        self.x = np.copy(X)
        self.y = np.copy(y)
        
        m, n = self.x.shape
        self.y = self.y.reshape((m, 1))
        
        self.init_param (n)
        self.optimize()

        yp = self.predict(self.x)
        accuracy = 100 - np.mean(np.abs(yp - self.y)) * 100
        
        print("\nAccuracy on training dataset = ", accuracy, "%")
        

    def predict (self, X):
        """
        Predict class labels for samples in X.
        """
        yhat = self.sigmoid(np.dot(X, self.w) + self.b)
        yp = np.round(yhat)
        return yp

    def score (self, X, y):
        """
        Function that computes the mean accuracy on the given test data X and labels y.

        """
        yp = self.predict(X)
        accuracy = 100 - np.mean(np.abs(yp - y.reshape(X.shape[0], 1))) * 100
        return accuracy
        
