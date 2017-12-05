"""
Author: Algan Rustinya
Description: Data preprocessing
"""
import csv
import numpy as np

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