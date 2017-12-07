import csv
import numpy as np
from collections import Counter

def extract_stock_prices(csv_file, symbol):
    matrix = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # remove header
        for row in reader:
            matrix.append(row)

    dictionary = dict()
    data = np.matrix(matrix)
    dictionary['time'] = data[:,0]
    dictionary['DJIA'] = data[:,1]
    dictionary['AAPL'] = data[:,2]
    dictionary['AMZN'] = data[:,3]
    dictionary['GOOG'] = data[:,4]
    dictionary['MSFT'] = data[:,5]

    dictionary['AAPL_label'] = data[:,7]
    dictionary['AMZN_label'] = data[:,8]
    dictionary['GOOG_label'] = data[:,9]
    dictionary['MSFT_label'] = data[:,10]

    res = np.hstack((dictionary['time'], dictionary['DJIA'], dictionary['{}'.format(symbol)], dictionary['{}_label'.format(symbol)]))

    return res

def extract_naive_bayes_data(csv_file):
    matrix = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # remove header
        for row in reader:
            matrix.append(row)

    data = np.array(matrix)
    timestamps = [e[:10] for e in data[:,0]]
    tweets = data[:,1]

    # Convert 'str1,str2,str3' to list[str1, str2, str3]
    tweets = [tweet.split(',') for tweet in tweets]

    try:
        sentiments = data[:,2]
    except IndexError:
        sentiments = []

    return timestamps, tweets, sentiments

def calculate_sentiment_percentages(timestamps, tweets, sentiments):
    unique_date = np.unique(timestamps)
    data = dict() # key = date, value = list of sentiments
    for date in unique_date:
        data[date] = []

    for date in unique_date:
        for i in range(len(timestamps)):
            # if date == timestamps[i] and int(sentiments[i]) != 0:
            if date == timestamps[i]:
                data[date].append(int(sentiments[i]))

    # calculate the percentage
    res = [] # [-1 percentage, 0 percentage, 1 percentage]
    for date in unique_date:
        sentiments = Counter(data[date]).most_common()
        percentages = []
        for sentiment in sentiments:
            label = sentiment[0]
            percentage = sentiment[1]/len(data[date])*100
            percentages.append((label, percentage))
        percentages = sorted(percentages)
        percentages = [tupple[1] for tupple in percentages]
        percentages = [date] + percentages
        res.append(percentages)
    return np.array(res)

def represent_data(symbol):
    djia_data = extract_stock_prices('raw_data/stock_prices.csv', symbol)
    timestamps, tweets, sentiments = extract_naive_bayes_data('naive_bayes_labeled_data/{}_tweets.csv'.format(symbol))
    sentiment_percentages = calculate_sentiment_percentages(timestamps, tweets, sentiments)

    dates = sentiment_percentages[:,0]
    djia_data = np.array(djia_data)
    djia_refined = []
    for i in range(len(djia_data)):
        if djia_data[i,0] in dates:
            djia_refined.append(djia_data[i])
    djia_refined = np.array(djia_refined)
    X = np.hstack((djia_refined[:,1:-1], sentiment_percentages[:,1:]))
    y = djia_refined[:,-1]

    # convert to float
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y

        
# if __name__ == '__main__':
#     X djia_stock, symbol_stock, -1_sentiment, 0_sentiment, 1_sentiment
#     X, y = represent_data('MSFT')

