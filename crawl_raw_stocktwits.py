"""
Author: Algan Rustinya
Description: Crawl Raw Stocktwits
"""

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
    """ Get the timestamp, body message and sentiment as dictionary
    """
    ret = []
    tweets = json_data['messages']
    last_id = None
    for tweet in tweets:
        relevant_data = dict()
        relevant_data['created_at'] = tweet['created_at']
        relevant_data['text'] = tweet['body'].replace('\n', ' ')
        try:
            relevant_data['sentiment'] = tweet['entities']['sentiment']['basic']
        except:
            relevant_data['sentiment'] = 'Neutral'
        ret.append(relevant_data)
        last_id = tweet['id']
        print(last_id)

    return ret, last_id

def create_csv(symbol, relevant_data):
    keys = relevant_data[0].keys()
    with open('stocktwits_training_data/%s_stocktwits.csv' % symbol, 'w') as f:
        dict_writer = csv.DictWriter(f, ['created_at', 'text', 'sentiment'])
        dict_writer.writeheader()
        dict_writer.writerows(relevant_data)
    f.close()

def get_stock_stream_wrapper(symbol, quantity):
    ret = []
    last_id = None
    while(quantity > 30):
        cur_ret, cur_id = get_relevant_data(get_stock_stream(symbol, params={'max':last_id}))
        if(cur_ret is None): break; #there are no more tweets to stream
        ret += cur_ret
        last_id = cur_id
        quantity -= 30
    cur_ret, cur_id = get_relevant_data(get_stock_stream(symbol, params={'limit':quantity, 'max':last_id}))
    ret += cur_ret
    return ret

if __name__ == '__main__':
    symbol = 'MSFT'
    relevant_data = get_stock_stream_wrapper(symbol, 3000)
    create_csv(symbol, relevant_data)