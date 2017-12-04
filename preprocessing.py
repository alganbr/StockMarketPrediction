"""
Author: Algan Rustinya
Description: Data preprocessing
"""

import csv
import re
import numpy as np
import nltk
# nltk.download()   # Uncomment this if this is your first time running nltk to download the corresponding packages
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def sentiment_to_label(sentiments):
    """
    Replace sentiments (Bullish, Bearish, Neutral) to labels (1, -1, 0)

    Parameters
    --------------------
        sentiments    -- numpy array of shape(n, )
                            n is the number of sentiments
    
    Returns
    --------------------
        labels      -- numpy array of shape (n, )
                        n is the number of labels

    """

    label_dict = {'Bearish': -1, 'Bullish': 1, 'Neutral': 0}
    labels = np.vectorize(label_dict.get)(sentiments)
    return labels

def preprocess_raw_text_data(csv_file):
    """
    Read from a csv file and return preprocessed text data (removal of stop words/punctuations and stemming)

    Parameters
    --------------------
        csv_file    -- csv file, file format: 'created_at', 'text'
    
    Returns
    --------------------
        timestamps  -- numpy array of shape (n, 1)
        tweets  -- numpy array of shape (n, d)
    """
    matrix = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # remove header
        for row in reader:
            matrix.append(row)

    data = np.array(matrix)
    timestamps = data[:,0]
    tweets = data[:,1]
    processed_tweets = []
    for tweet in tweets:
        processed_tweets.append(extract_words(tweet))
    return timestamps, processed_tweets

def extract_data(csv_file):
    """
    Read from a csv file and return array of tweets and setiment labels

    Parameters
    --------------------
        csv_file    -- csv file, file format: 'created_at', 'text', 'likes', 'sentiment'
    
    Returns
    --------------------
        timestamps  -- numpy array of shape (n, d)
        tweets      -- numpy array of shape (n, d) 
                        n is the number of tweets,
                        d is the length of the tweet
        labels      -- numpy array of shape (n, )
                        n is the number of labels

    """

    matrix = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # remove header
        for row in reader:
            matrix.append(row)

    data = np.array(matrix)
    timestamps = data[:,0]
    tweets = data[:,1]
    likes = data[:,2]
    sentiments = data[:,3]
    labels = sentiment_to_label(sentiments)

    return timestamps, tweets, likes, labels

def extract_words(input_string):
    """
    Processes the input_string. Remove all URLs, tokenize words 
    with nltk word tokenizer and perform stemming.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    # Remove URLs
    input_string = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', input_string)

    # Tokenize words
    word_list = word_tokenize(input_string.lower())

    # Stemming
    stemmer = PorterStemmer()
    word_list = [stemmer.stem(word) for word in word_list]

    # Remove stop words and punctuations
    stop_words = set(stopwords.words('english'))
    word_list = [word for word in word_list if word not in punctuation and word not in stop_words]
    return word_list


def extract_dictionary(tweets):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        tweets    -- numpy array of size (n, d)
                        n is the number of tweets,
                        d is the length of the tweet
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    words = []
    for tweet in tweets:
        words += extract_words(tweet)
    unique_words, counts = np.unique(words, return_counts=True)
    word_list = dict(zip(unique_words, counts))
    return word_list

def extract_feature_vectors(tweets, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        tweets         -- numpy array of size (n, d)
                            n is the number of tweets,
                            d is the length of the tweet
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = len(tweets)
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    for i in range(num_lines):
        tweet = extract_words(tweets[i])
        for j in range(num_words):
            if list(word_list.keys())[j] in tweet:
                feature_matrix[i][j] = 1
        
    return feature_matrix