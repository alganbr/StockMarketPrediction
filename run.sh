#!/bin/bash
echo ""
echo "Running Naive Bayes classifier to label tweets"
echo ""
python3 ./naive_bayes.py
echo "--------------------------------------------"
echo ""
echo "Running logistic regression for stock price prediction model"
echo ""
python3 ./logistic_regression.py
echo "--------------------------------------------"
echo ""
echo "Running SVM for stock price prediction model"
echo ""
python3 ./svm.py AAPL
python3 ./svm.py GOOG
python3 ./svm.py MSFT

