#!/bin/bash

python ./preprocessing_raw_data.py
python ./naive_bayes.py
python ./logistic_regression.py
python ./svm.py AAPL
python ./svm.py MSFT

