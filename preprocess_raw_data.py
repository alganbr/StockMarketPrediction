# preprocess_raw_tweets
# Created by Elaine Lin 12/4/2017
# ---------------------------------

import preprocessing as pp
import csv
import numpy as np
import glob
import datetime

class Preprocess_Raw_Tweets():

    def __init__(self):
        self.directories = ['raw_data/AAPL_tweets', 'raw_data/GOOG_tweets', 'raw_data/MSFT_tweets', 'raw_data/AMZN_tweets']
        self.stocknames = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
        # self.directories = ['stocktwits_training_data']
        # self.time_format = "%Y-%m-%dT%H:%M:%SZ"    # this is the time_format for stocktwits tweets
        self.time_format = "%Y-%m-%d %H:%M:%S"       # this is the time_format for tweeter tweets
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
            processed_tweets = self.extract_words(sorted_matrix[:, 1])
            self.write_to_csv(self.stocknames[csv_files.index(file_list)], created_at, processed_tweets)

    def extract_words(self, raw_tweets):
        return pp.preprocess_raw_text_data(raw_tweets)

    def write_to_csv(self, stockname, created_at, processed_tweets):
        # Write the csv
        with open('preprocessed_data/%s_tweets.csv' % stockname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['created_at', 'text'])
            writer.writerows(np.transpose(np.vstack((created_at, processed_tweets))))
        pass

if __name__ == '__main__':
    # matrix = []
    # with open('stocktwits_training_data/AMZN_stocktwits.csv', 'r') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     next(reader) # remove header
    #     for row in reader:
    #         matrix.append(row)

    # data = np.array(matrix)
    # created_at = data[:,0]
    # tweets = data[:,1]
    # tweets = pp.preprocess_raw_text_data(tweets)

    prt = Preprocess_Raw_Tweets()
    prt.preprocess_entries()
    pass