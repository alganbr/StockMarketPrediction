"""
Author: Algan Rustinya
Description: Logistic Regression
"""

import math
import numpy as np
import preprocessing as pp

class LogisticRegression():

    def __init__(self):
        self.coef = None

    def sigmoid(self, scores):
        return float(1)/(1 + np.exp(-scores))

    def log_likelihood(self, X, y):
        scores = np.dot(X, self.coef)
        ll = np.sum(y * scores - np.log(1 + np.exp(scores)))
        return ll

    def predict(self, X):
        scores = np.dot(X, self.coef)
        y_pred = self.sigmoid(scores)
        return y_pred

    def gradient(self, X, y):
        y_pred = self.predict(X)
        gradient = np.dot(X.T, np.subtract(y, y_pred))
        return gradient

    def error(self, X, y): # calculate error using residual sum of squares (RSS)
        n,d  = X.shape
        y_pred = self.predict(X)
        err =  np.sum(np.power(y - y_pred, 2)) / float(n)
        return err

    def generate_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        return X

    def fit(self, X, y, tmax=10000, eta=0.01, eps=0):
        X = self.generate_intercept(X)
        n,d = X.shape
        self.coef = np.zeros(d)
        err_list  = np.zeros((tmax,1)) 

        for t in range(tmax):
            # update coefficient using gradient ascent
            self.coef += eta * self.gradient(X, y)
            err_list[t] = self.error(X,y)

            # stop condition
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break

        return self

if __name__ == '__main__':
    timestamps, tweets, labels = pp.extract_data('stocktwits_training_data/AMZN_stocktwits/AMZN_stocktwits.csv')
    dictionary = pp.extract_dictionary(tweets)
    X = pp.extract_feature_vectors(tweets, dictionary)
    y = labels

    model = LogisticRegression()
    model.fit(X, y, eta=0.0001)
    print(model.coef)