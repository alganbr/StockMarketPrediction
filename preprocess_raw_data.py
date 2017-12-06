# preprocess_raw_tweets
# Created by Elaine Lin 12/4/2017
# ---------------------------------

import csv
import re
import numpy as np
import nltk
# nltk.download()   # Uncomment this if this is your first time running nltk to download the corresponding packages
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import datetime

class Preprocess_Raw_Tweets():

    def __init__(self, is_stocktwits):
        self.directories = ['raw_data/AAPL_tweets', 'raw_data/GOOG_tweets', 'raw_data/MSFT_tweets', 'raw_data/AMZN_tweets'] if not is_stocktwits else ['stocktwits_training_data/AAPL_stocktwits', 'stocktwits_training_data/GOOG_stocktwits', 'stocktwits_training_data/AMZN_stocktwits']
        self.stocknames = ['AAPL', 'GOOG', 'AMZN']
        self.time_format = "%Y-%m-%d %H:%M:%S" if not is_stocktwits else "%Y-%m-%dT%H:%M:%SZ"
        self.literals = self.prepare_literals()
        self.is_stocktwits = is_stocktwits

    def prepare_literals(self):
        literals = ['\\xe2', '\\x9a', '\\x9b', '\\x9c', '\\x9d', '\\x9e', '\\x9f', '\\x8a', '\\x8b', '\\x8c', '\\x8d', '\\x8e',
            '\\x8f', '\\xf0', '\\xc2']
        for i in range(80, 100):
            literals.append('\\x' + str(i))
        for i in range(0, 10):
            literals.append('\\xa' + str(i))
            literals.append('\\xb' + str(i))
        return literals

    def preprocess_entries(self):
        csv_files = []  # List of file names in each directory
        for directory in self.directories:
            csv_files.append(glob.glob(directory + '/*.csv'))
        for file_list in csv_files:
            matrix = [] # Data matrix that will store all tweets related to one stock
            for file in file_list:
                with open(file, 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    next(reader)
                    for row in reader:
                        matrix.append(row)
    
            # Avoid duplicate entries
            matrix = np.vstack({tuple(row) for row in matrix})
            
            # Sort entries
            sorted_matrix = np.array(sorted(matrix, key=lambda matrix: datetime.datetime.strptime(matrix[0],self.time_format)))
            
            if self.is_stocktwits is False:
                # Remove entries created on dates when market is closed or not in the month of Nov, 2017
                indices_to_remove = []
                for ind in range(0,sorted_matrix.shape[0]):
                    entry = sorted_matrix[ind]
                    date = datetime.datetime.strptime(entry[0],self.time_format)
                    if date.month != 11 or date.day == 23 or date.weekday() in [5, 6]:
                        indices_to_remove.append(ind)
                sorted_matrix = np.delete(sorted_matrix, indices_to_remove, 0)

            # Write to csv
            created_at = sorted_matrix[:,0]
            processed_tweets = self.preprocess_raw_text_data(sorted_matrix[:, 1])
            self.write_to_csv(self.stocknames[csv_files.index(file_list)], created_at, processed_tweets) if not self.is_stocktwits else self.write_to_csv(self.stocknames[csv_files.index(file_list)], created_at, processed_tweets, sorted_matrix[:,2])

    def preprocess_raw_text_data(self, tweets):
        """
        Read from a csv file and return preprocessed text data (removal of stop words/punctuations and stemming)

        Parameters
        --------------------
            tweets    -- raw tweets
        
        Returns
        --------------------
            processed_tweets  -- numpy array of shape (n, d)
        """
        processed_tweets = []
        for tweet in tweets:
            processed_tweets.append(self.extract_words(tweet))
        return processed_tweets

    def extract_words(self, input_string):
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
        # Remove linebreaks and initial ' b' ' in tweets
        input_string = input_string[2:].replace('\\n', ' ')

        # Remove URLs
        input_string = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', input_string)

        # Remove common unicode literals
        for literal in self.literals:
            input_string = input_string.replace(literal, ' ')

        # Tokenize words
        word_list = word_tokenize(input_string.lower())

        # Stemming
        stemmer = PorterStemmer()
        for ind in range(0, len(word_list)):
            word = word_list[ind]
            try:
                word_list[ind] = stemmer.stem(word)
            except IndexError:
                print('error', word)
        # Remove stop words and punctuations
        stop_words = set(stopwords.words('english'))
        word_list = [word for word in word_list if word != "" and word not in punctuation and word not in stop_words]
        return word_list

    def write_to_csv(self, stockname, created_at, processed_tweets, sentiment):
        csv_name = 'preprocessed_data/%s_stocktwits.csv' if self.is_stocktwits else 'preprocessed_data/%s_tweets.csv'
        # Write the csv
        with open(csv_name % stockname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['created_at', 'text']) if not self.is_stocktwits else writer.writerow(['created_at', 'text', 'sentiment'])
            writer.writerows(np.transpose(np.vstack((created_at, processed_tweets)))) if not self.is_stocktwits else writer.writerows(np.transpose(np.vstack((created_at, processed_tweets, sentiment))))
        pass

if __name__ == '__main__':
    prt = Preprocess_Raw_Tweets(True)
    prt.preprocess_entries()