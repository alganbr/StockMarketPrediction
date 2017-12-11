# Created by Elaine Lin, 10/30/2017

import tweepy
import csv

# Change the following
consumer_key="qjI4UBn8AIkGl9ZsrdW5UuOnK"
consumer_secret="ttAT7Lk70ag4HbxT61P6lPPk8TumrqyrkLgTGFPJRlxBCafHl0"
access_token="919342793415319552-Ahj0eQVaKE0jBZ5vfuJvunHsvc9IMyy"
access_token_secret="4ICIlUHAmNc7JYJNDNaRk8bmzwWG3aRyDDYZkmtrVplNe"

query = "AAPL"

######################

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

out_tweets = []

try:
    for page in tweepy.Cursor(api.search,
                           q=query,
                           count=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").pages(10000):
        # Transform the tweepy tweets into a 2D array that will populate the CSV
        out_tweets.extend([[tweet.created_at, tweet.text.encode("utf-8")] for tweet in page])
except tweepy.error.TweepError as err:
    print(err.reason, err.response)
    print('Max limit reached')

# Write the csv
with open('../data/raw_data/%s_tweets.csv' % query, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['created_at', 'text'])
    writer.writerows(out_tweets)
pass
