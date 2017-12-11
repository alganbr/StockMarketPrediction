{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf100
{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs20 \cf0 README.TXT\
\
- crawl_raw_stocktwits.py\
\
Crawl tweets from StockTwits and save to data/stocktwits_training_data\
\
- crawl_raw_tweets.py\
\
Crawl tweets from Tweeter and save to data/raw_data\
\
- preprocess_raw_data.py\
\
Perform text preprocessing to turn tweet text into bag of words. Save results to data/preprocessed_data\
\
- naive_bayes.py\
\
Train Naive Bayes classifier from Stocktwits tweets to label Tweeter tweets and save results to data/naive_bayes_labeled_data\
\
- data_preparation.py\
\
Prepare data for logistic regression and svm\
\
- logistic_regression.py\
\
Perform logistic regression for stock price prediction\
\
- svm.py\
\
Perform sum for stock price prediction\
\
- run.sh\
\
Shell script to run the project}