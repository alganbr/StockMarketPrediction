"""
Author: Algan Rustinya
Description: Logistic Regression
"""

import math
import numpy as np
import data_preparation as dp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore') # ignore warning for metric evaluation 

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

    def fit(self, X, y, tmax=10000, eta=0.00001, eps=0):
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

######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    # map continuous-valued predictions to binary labels
    y_label = y_pred
    y_label[y_label>=0.5] = 1
    y_label[y_label<0.5] = 0
    
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)
    elif metric == "sensitivity":
        cm = metrics.confusion_matrix(y_true, y_label)
        return float(cm[1][1])/(cm[1][1] + cm[1][0])
    elif metric == "specificity":
        cm = metrics.confusion_matrix(y_true, y_label)
        return float(cm[0][0])/(cm[0][0] + cm[0][1])

    return None

def cv_performance(clf, X, y, kf, metric="accuracy"):        
    perf_list = []
    for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        X_test = clf.generate_intercept(X_test)
        y_pred = clf.predict(X_test)
        perf = performance(y_true=y_test, y_pred=y_pred, metric=metric)
        perf_list.append(perf)

    res = np.mean(np.array(perf_list))
    print("metric: {0}, score: {1}".format(metric, res))

    return res

def performance_test(clf, X, y, metric="accuracy"):
    X = clf.generate_intercept(X)
    y_pred = clf.predict(X)
    score = performance(y_true=y, y_pred=y_pred, metric=metric)     
    print("    metric: {0}, score: {1:.4f}".format(metric, score))
    return score

if __name__ == '__main__':
    symbols = ['AAPL', 'GOOG', 'MSFT']
    for symbol in symbols:
        print("Evaluating stock prediction for {}".format(symbol))
        X, y = dp.represent_data(symbol)
        split = int(len(X)/10 * 8) # 80% training data 20% test data
        clf = LogisticRegression()

        # evaluation
        kf = StratifiedKFold(y=y, n_folds=5)
        metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
        for metric in metric_list:
            cv_performance(clf=clf, X=X, y=y, kf=kf, metric=metric)
        print("")
