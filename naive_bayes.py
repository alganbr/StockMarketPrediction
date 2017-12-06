# naive_bayes.py
# Created by Elaine Lin 12/5/17

import numpy as np
import csv
from sklearn.model_selection import train_test_split

class NaiveBayes():
    def __init__(self):
        self.beta = []
        self.pi = []
        self.vocabulary = []
        self.unique_classes = []

    def train_classifer(self, csv_file):
        # Train the classifer based on training sets in csv_file

        # Read data
        _, tweets, sentiments = self.extract_data(csv_file)

        # Convert 'str1,str2,str3' to list[str1, str2, str3]
        tweets = [tweet.split(',') for tweet in tweets]

        # Collect vocabulary of all documents
        self.vocabulary = list(self.extract_vocabulary(tweets))

        
        tweets_train, tweets_test = 
        # Count number of classes (sentiments)
        self.unique_classes, class_counts = np.unique(sentiments, return_counts=True)

        # Calculate pi values for classifier for each class
        self.pi = [count/np.sum(class_counts) for count in class_counts]

        # Sort data based on class
        data_matrix = np.vstack((tweets, sentiments))
        sorted_matrix_list = []
        for ind in range(0, len(self.unique_classes)):
            indices = np.where(data_matrix[1,:] == self.unique_classes[ind])
            sorted_matrix_list.append(data_matrix[:,indices])
    
        # Calculate number of each unique word for each class
        counts_of_unique_word = np.zeros([len(self.unique_classes), len(self.vocabulary)])
        for i in range(0, len(sorted_matrix_list)):
            doc_set = sorted_matrix_list[i]
            flattened_doc = [word for sublist in doc_set[0] for sublist2 in sublist for word in sublist2]
            unique_words, word_counts = np.unique(flattened_doc, return_counts=True)
            for j in range(0, len(unique_words)):
                counts_of_unique_word[i, self.vocabulary.index(unique_words[j])] = word_counts[j]

        # Calculate beta values for every word in vocabulary
        self.beta = np.zeros([len(self.unique_classes), len(self.vocabulary)])
        for c in range(0, len(self.unique_classes)):
            for ind in range(0, len(self.vocabulary)):
                self.beta[c, ind] = (counts_of_unique_word[c, ind] + 1)/(np.sum(counts_of_unique_word[c,:]) + len(self.vocabulary))

    def cross_validation(self, cv_data):


    def predict(self, csv_file):
        # Use the trained classifer to predict class of a given doc

        # Read data
        timestamps, tweets, _ = self.extract_data(csv_file)

        # Convert 'str1,str2,str3,...' to list[str1, str2, str3,...]
        tweets = [tweet.split(',') for tweet in tweets]

        # Calculate likelihood
        likelihood = np.zeros([len(tweets), len(self.unique_classes)])
        for t in range(0, len(tweets)):
            tweet = tweets[t]
            likelihood[t, :] = self.pi
            for word in tweet:
                ind = self.vocabulary.index(word)
                for c in range(0, len(self.unique_classes)):
                    likelihood[t, c] *= self.beta[c, ind]
        sentiments = self.unique_classes[np.argmax(likelihood, axis=1)]

    def extract_vocabulary(self, tweets):
        """
        Given a filename, reads the text file and builds a dictionary of unique
        words/punctuations.
        
        Parameters
        --------------------
            tweets    -- numpy array of size (n, d)
                            n is the number of tweets,
                            where each tweet is list of strings separated by ',',
                            d is the length of the tweet
        
        Returns
        --------------------
            word_list -- set of words (vocabulary)
        """

        word_list = {word for tweet in tweets for word in tweet}
        return word_list

    def extract_data(self, csv_file):
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
            sentiments      -- numpy array of shape (n, )
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
        try:
            sentiments = data[:,2]
        except IndexError:
            sentiments = []

        return timestamps, tweets, sentiments

if __name__ == '__main__':
    model = NaiveBayes()
    model.train_classifer('preprocessed_data/naive_bayes_training_set.csv')
    model.predict('preprocessed_data/naive_bayes_test_set.csv')
