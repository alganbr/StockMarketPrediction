# stocktwits_api.py
# ------------------
# Save classified tweets from stocktwits.com to csv file
#

import os
import requests
import json
import urllib
import logging as log
import csv

consumer_key="e109a53b775bb9c4"
consumer_secret="eae3a518ba2b58a60eaec81fdc606a5ddd049a83"
access_token="f34876572e42e7157fdcd7e8143ec0079e857f4e"

# StockTwits details
ST_BASE_URL = 'https://api.stocktwits.com/api/2/'
ST_BASE_PARAMS = dict(access_token='f34876572e42e7157fdcd7e8143ec0079e857f4e')

# ---------------------------------------------------------------------
# Requestor
# ---------------------------------------------------------------------

def get_json(url, params=None):
    """ Uses tries to GET a few times before giving up if a timeout.  returns JSON
    """
    resp = None
    for i in range(4):
        try:
            resp = requests.get(url, params=params, timeout=5)
        except requests.Timeout:
            trimmed_params = {k: v for k, v in params.items() if k not in ST_BASE_PARAMS.keys()}
            log.error('GET Timeout to {} w/ {}'.format(url[len(ST_BASE_URL):], trimmed_params))
        if resp is not None:
            break
    if resp is None:
        log.error('GET loop Timeout')
        return None
    else:
        return json.loads(resp.content)

# ---------------------------------------------------------------------
# Basic StockTwits interface
# ---------------------------------------------------------------------
def get_stock_stream(symbol, params={}):
    """ gets stream of messages for given symbol
    """
    all_params = ST_BASE_PARAMS.copy()
    for k, v in params.items():
        all_params[k] = v
    return get_json(ST_BASE_URL + 'streams/symbol/{}.json'.format(symbol), params=all_params)

def get_trending_stocks():
    """ returns list of trending stock symbols
    """
    trending = get_json(ST_BASE_URL + 'trending/symbols.json', params=ST_BASE_PARAMS)['symbols']
    symbols = [s['symbol'] for s in trending]
    return symbols

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def get_relevant_data(json_data):
    """ Get the timestamp, body message, number of likes, number of reshares, and sentiment as dictionary
    """
    ret = []
    tweets = json_data['messages']
    for tweet in tweets:
        relevant_data = dict()
        relevant_data['created_at'] = tweet['created_at']
        relevant_data['text'] = tweet['body']
        try:
            relevant_data['likes'] = tweet['likes']['total']
        except:
            relevant_data['likes'] = 0
        try:
            relevant_data['sentiment'] = tweet['entities']['sentiment']['basic']
        except:
            relevant_data['sentiment'] = 'Neutral'
        ret.append(relevant_data)

    return ret

def create_csv(symbol, relevant_data):
    keys = relevant_data[0].keys()
    with open('%s_stocktwits.csv' % symbol, 'w') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(relevant_data)

if __name__ == '__main__':
    symbol = 'AMZN'
    relevant_data = get_relevant_data(get_stock_stream(symbol))
    create_csv(symbol, relevant_data)