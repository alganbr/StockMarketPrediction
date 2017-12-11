import math
import numpy
import data_preparation
from sklearn import svm
from sklearn import metrics
import sys
import warnings

warnings.filterwarnings('ignore') # ignore warning for metric evaluation 

class SVM():
	def __init__(self,X,y):
		self.svm = svm.SVC()
		self.svm.fit(X,y)

	def fit(self, X, y):
		self.svm.fit(X,y)

	def predict(self, X):
		return self.svm.predict(X)

	def get_params():
		return self.svm.get_params()
	
	def decision_function(X):
		return self.svm.decision_function(X)
			

def sentiment_sum(timestamps, sentiments):
	unique_date = numpy.unique(timestamps)
	data = dict() # key = date, value = list of sentiments
	for date in unique_date:
		data[date] = [0,0,0]
	for date in unique_date:
		for i in range(len(timestamps)):
			if date == timestamps[i]:
				temp = data[date]
				if sentiments[i] == '-1':
					temp[0] += 1
					
				if sentiments[i] == '0':
					temp[1] += 1
				if sentiments[i] == '1':
					temp[2] += 1
				data[date] = temp
	return data

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print("No argument given, stock code required");
		exit()
	symbol = sys.argv[1]
	djia_data = data_preparation.extract_stock_prices('raw_data/stock_prices.csv', symbol)
	timestamps, tweets, sentiments = data_preparation.extract_naive_bayes_data('naive_bayes_labeled_data/{}_tweets.csv'.format(symbol))
	sentiment_totals = sentiment_sum(timestamps, sentiments)
	
	djia_data = numpy.array(djia_data)
	dates = sentiment_totals.keys()
	dates.sort()
	djia_refined = []
	for i in range(len(djia_data)):
		if djia_data[i,0] in dates:
			djia_refined.append(djia_data[i])
	djia_refined = numpy.array(djia_refined)
	
	temp = []
	for date in dates:
		temp.append(sentiment_totals[date])
	X = numpy.hstack((djia_refined[:,1:-1], temp))
	y = djia_refined[:,-1]

	X = numpy.array(X, dtype=float)
	y = numpy.array(y, dtype=float)
	
	split = int(len(X)/10 * 8) # 80% training data 20% test data
	if split == 0:
		split += 1
	clf = SVM(X[0:split],y[0:split])
	
	y_true = y[split+1:]
	y_label = clf.predict(X[split+1:])
	
	print("Evaluating stock prediction for {:s}".format(symbol))
	print("accuracy: {:f}".format(metrics.accuracy_score(y_true, y_label)))
	print("f1_score: {:f}".format(metrics.f1_score(y_true, y_label)))
	print("auroc: {:f}".format(metrics.roc_auc_score(y_true, y_label)))
	print("precision: {:f}".format(metrics.precision_score(y_true, y_label)))
	cm = metrics.confusion_matrix(y_true, y_label)
	print("sensitivity: {:f}".format(float(cm[1][1])/(cm[1][1] + cm[1][0])))
	cm = metrics.confusion_matrix(y_true, y_label)
	print("specificity: {:f}".format(float(cm[0][0])/(cm[0][0] + cm[0][1])))
