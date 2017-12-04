import tweepy

from textblob import TextBlob

consumer_key="Q97m94t5iPuV75dn5CJVeUCWu"
consumer_secret="kdbe49xaaUFka0jEGPJgqnWGEnaPQpHVSygQeoM6oHilV2A3s2"
access_token="210774490-dUNyIchnOTC0XZCHLUMhALUgKErWlL7RyHb473xQ"
access_token_secret="xzPwyDaevrslC3fwC4LEGZ6zLtd9aoieybXXRGhaY9Nnl"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def stock_sentiment(quote, num_tweets):
    list_of_tweets = tweepy.Cursor(api.search,
                           q=quote,
                           count=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items(100)
    positive, negative, neutral = 0, 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment

        if blob.polarity > 0:
            positive += 1
        if blob.polarity < 0:
        	negative += 1
        if blob.polarity == 0:
        	neutral += 1

        print(tweet.text)
        print(blob.polarity)


    return positive, negative, neutral

if __name__ == '__main__':
	stock = 'AAPL'
	positive, negative, neutral = stock_sentiment(stock, 10000)
	print("{0}, {1}, {2}".format(positive, negative, neutral))