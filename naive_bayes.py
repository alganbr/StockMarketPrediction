# naive_bayes.py
# Created by Elaine Lin 12/5/17

import csv
import numpy as np
from sklearn.model_selection import KFold

class NaiveBayes():
    def __init__(self):
        self.beta = []
        self.pi = []
        self.vocabulary = []
        self.unique_classes = []
        self.kf_split_indices = []
        self.k_fold_num = 10

    def prepare_cross_validation(self, sorted_data_matrix):
        """
        Split sorted (by class) training data matrix into k-fold, k == self.k_fold_num
        Using sorted data because want to evenly distribute each class in each training subset.
        
        """       
        # Split data into kfold
        kf = KFold(n_splits=self.k_fold_num, shuffle=True)
        self.kf_split_indices = np.empty([len(self.unique_classes), 10], dtype=object)

        for c in range(0, len(self.unique_classes)):
            sorted_data = np.array(sorted_data_matrix[c][0,:]).transpose()
            # k-fold split each class because want to make sure number of each classes are evenly distributed
            k_iter = 0
            for train_index, test_index in kf.split(sorted_data):
                self.kf_split_indices[c, k_iter] = [train_index, test_index]
                k_iter += 1

    def sort_data_matrix(self, data_matrix):
        """
        Input
            - data_matrix: 2 x n matrix, where n is the number of examples,
                and class labels are in the second row
        Output
            - sorted_matrix_list: list of 2 x m matrix, where m is the number
                of examples in each corresponding class
        """

        # Sort data based on class
        sorted_matrix_list = []
        for ind in range(0, len(self.unique_classes)):
            indices = np.where(data_matrix[1,:] == self.unique_classes[ind])
            sorted_matrix_list.append(data_matrix[:,indices])
        return sorted_matrix_list

    def calculate_model_coeff(self, training_set_list):
        """
        Input
            - training_set_list: c x n matrix, where c is the number of unique classes, and n is the number of documents

        Output
            - beta: c x m matrix where c is the number of unique classes and m is the number of vocabulary
                This beta could be different from self.beta, since beta is derived from the specified training_set_list,
                whereas self.beta will eventually become the average beta from all training subsets.
        """
        # Calculate number of each unique word for each class
        counts_of_unique_word = np.zeros([len(self.unique_classes), len(self.vocabulary)])
        for i in range(0, len(self.unique_classes)):
            doc_set = training_set_list[0,i]
            flattened_doc = [word for sublist in doc_set for word in sublist]
            unique_words, word_counts = np.unique(flattened_doc, return_counts=True)
            for j in range(0, len(unique_words)):
                counts_of_unique_word[i, self.vocabulary.index(unique_words[j])] = word_counts[j]

        beta = np.zeros([len(self.unique_classes), len(self.vocabulary)])
        # Calculate beta values for every word in vocabulary
        for c in range(0, len(self.unique_classes)):
            for ind in range(0, len(self.vocabulary)):
                self.beta[c, ind] += (counts_of_unique_word[c, ind] + 1)/(np.sum(counts_of_unique_word[c,:]) + len(self.vocabulary))
                beta[c, ind] += (counts_of_unique_word[c, ind] + 1)/(np.sum(counts_of_unique_word[c,:]) + len(self.vocabulary)) # Local copy for cv purpose
        return beta

    def train_classifer(self, training_set_file):
        """
        Train text classifier based on a training_set_file
        Input
            - training_set_file: a csv file containing [timestamps, documents, classes]
        """

        # Read data
        _, tweets, sentiments = self.extract_data(training_set_file)

        # Collect vocabulary of all documents
        self.vocabulary = list(self.extract_vocabulary(tweets))

        # Count number of classes (sentiments)
        self.unique_classes, class_counts = np.unique(sentiments, return_counts=True)

        # Calculate pi values for classifier for each class
        self.pi = [count/np.sum(class_counts) for count in class_counts]

        # Sort data based on class
        data_matrix = np.vstack((tweets, sentiments))
        sorted_matrix_list = self.sort_data_matrix(data_matrix)

        # Prepare for cross_validation
        self.prepare_cross_validation(sorted_matrix_list)

        # Prepare beta
        self.beta = np.zeros([len(self.unique_classes), len(self.vocabulary)])

        # Iterate through each fold to calculate beta values and cv
        for k_iter in range(0, self.kf_split_indices.shape[1]):
            training_set_x = []
            training_set_y = []
            testing_set_x = []
            testing_set_y = []
            for c in range(0, len(self.unique_classes)):
                train_index, test_index = self.kf_split_indices[c][k_iter]
                training_set_x.append(sorted_matrix_list[c][0][0][train_index])
                training_set_y.append(sorted_matrix_list[c][1][0][train_index])
                testing_set_x.extend(sorted_matrix_list[c][0][0][test_index])
                testing_set_y.extend(sorted_matrix_list[c][1][0][test_index])
            training_set = np.vstack((training_set_x, training_set_y))
            testing_set = np.vstack((testing_set_x, testing_set_y))
            
            beta = self.calculate_model_coeff(training_set)
            self.calculate_cross_validation_error(testing_set, beta)
        
        # Finally, average self.beta values from all k folds
        self.beta = np.array(self.beta, dtype='f')
        self.beta = self.beta/self.k_fold_num
        
        # Misclassification error for the entire training set 
        # (this error should be discarded, as it does not represent the true misclassification error for this classifier)
        # But in case you are curious:
        # self.calculate_cross_validation_error(data_matrix, self.beta)

    def calculate_cross_validation_error(self, testing_set, beta):
        """
        Calculate misclassification error in each k-fold iteration, using the beta values specified

        """
        # Calculate log likelihood
        sentiments = self.calculate_likelihood(testing_set[0,:], beta)
        ground_truths = list(map(int, testing_set[1,:]))
        subtraction_result = np.subtract(np.array(sentiments), np.array(ground_truths))
        misclassification_error = np.count_nonzero(subtraction_result)/testing_set.shape[1]
        print('Misclassification Error with k-fold cv:', misclassification_error)

    def calculate_likelihood(self, docs, beta):
        """
        Calculate log likelihood and return class with highest likelihood for the corresponding document,
        using the specified beta values.

        Input
            - docs : lists of lists of words, e.g. [['word_1', 'word_2',...], [...], ...]
            - beta : beta values to calculate likelihood, 
                    n x m matrix where n is the number of unique classes of docs, 
                    and m is the number of vocabulary
        
        Output
            - classes: Class prediction for the corresponding document. List of type int.
        """
        # Calculate likelihood
        likelihood = np.zeros([len(docs), len(self.unique_classes)])
        for t in range(0, len(docs)):
            tweet = docs[t]
            likelihood[t, :] = self.pi
            for word in tweet:
                ind = self.vocabulary.index(word)
                for c in range(0, len(self.unique_classes)):
                    likelihood[t, c] *= beta[c, ind]
        likelihood = np.log10(likelihood)
        classes = self.unique_classes[np.argmax(likelihood, axis=1)]
        return list(map(int, classes))

    def predict(self, csv_file):
        """
        Use the trained classifer to predict class of a given csv_file
        """

        # Read data
        timestamps, tweets, _ = self.extract_data(csv_file)

        # Calculate sentiments
        sentiments = self.calculate_likelihood(tweets, self.beta)

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

        # Convert 'str1,str2,str3' to list[str1, str2, str3]
        tweets = [tweet.split(',') for tweet in tweets]

        try:
            sentiments = data[:,2]
        except IndexError:
            sentiments = []

        return timestamps, tweets, sentiments

if __name__ == '__main__':
    model = NaiveBayes()
    model.train_classifer('preprocessed_data/GOOG_stocktwits.csv')
    # model.predict('preprocessed_data/naive_bayes_test_set.csv')
