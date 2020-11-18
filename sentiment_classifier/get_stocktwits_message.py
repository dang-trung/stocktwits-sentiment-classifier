#!/usr/bin/env python3
"""StockTwits Messages Downloader.

    Download messages from StockTwits API using multiple access keys.
"""
import csv
import json
import os
import time
import pandas as pd
import requests
from requests.exceptions import HTTPError


def get_stocktwits_message(symbol, start="", end="",
                           file_name='data/stocktwits.csv'):
    """
    Download messages from StockTwits API using multiple access keys.
    Write those messages to a .csv file.

    Parameters
    ----------
    symbol : str
        StockTwits cashtags to be retrieved messages.
        e.g. Bitcoin has the cashtag "$BTC.X".
        Find more cashtags in the file at "data/symbols.csv"
    start : str
        Start date of retrieved messages (YYYY-MM-DD)
    end :
        End date of retrieved messages (YYYY-MM-DD)
    file_name : str
        Path of the file to be written in (defaults to data/stocktwits.csv)

    Returns
    -------
    NoneType
        None

    """
    print(f"Getting messages with cashtags {symbol}...")
    fields = ['symbol', 'message', 'sentiment', 'datetime', 'user',
              'message_id']
    token = 0
    base_url = 'https://api.stocktwits.com/api/2/streams/symbol/'

    access_token = ['',
                    'access_token=32a3552d31b92be5d2a3sd282ca3a864f96e95818&',
                    'access_token=44ae93a5279092f7804a0ee04753252cbf2ddfee&',
                    'access_token=990183ef04060336a46a80aa287f774a9d604f9c&']

    file = open(file_name, 'a', newline='', encoding='utf-8')
    start_date = pd.Timestamp(start, tz='UTC')
    end_date = pd.Timestamp(end, tz='UTC')

    # Determine where to start if resuming script
    if os.stat(file_name).st_size == 0:
        # Open file in append mode and write headers to file
        last_message_id = None
        csvfile = csv.DictWriter(file, fields)
        csvfile.writeheader()
    else:
        # First extract last message id then open file in append mode
        # without writing headers
        file = open(file_name, 'r', newline='', encoding='utf-8')
        csvfile = csv.DictReader((line.replace('\0', '') for line in file))
        data = list(csvfile)
        data = data[-1]
        last_message_id = data['message_id']
        file.close()
        file = open(file_name, 'a', newline='', encoding='utf-8')
        csvfile = csv.DictWriter(file, fields)

    stocktwit_url = (base_url + symbol + ".json?" + access_token[token])
    if last_message_id is not None:
        stocktwit_url += "max=" + str(last_message_id)

    api_hits = 0
    while True:
        try:
            response = requests.get(stocktwit_url)
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            response = None
        except Exception as err:
            print(f'Other error occurred: {err}')
            response = None

        if response is not None:

            if response.status_code == 429:
                time_left = int(response.headers['X-RateLimit-Reset']) - int(
                    time.time())
                print(
                    f"REQUEST IP RATE LIMITED FOR "
                    f"{time_left} seconds."
                )

                if not response.status_code == 200:
                    stocktwit_url = (
                            base_url + symbol + ".json?" + access_token[token]
                            + "max=" + str(last_message_id))
                token = (token + 1) % (len(access_token))
                continue

            reach_start_date = False
            api_hits += 1
            response = json.loads(response.text)
            last_message_id = response['cursor']['max']
            # Write data to csv file
            for message in response['messages']:
                # Prepare object to write in csv file
                if pd.Timestamp(message['created_at']) < start_date:
                    reach_start_date = True
                elif pd.Timestamp(message['created_at']) > end_date:
                    pass
                else:
                    obj = {
                        'symbol': symbol, 'message': message['body'],
                        'datetime': message['created_at'],
                        'user': message['user']['id'],
                        'message_id': message['id']
                    }
                    if message['entities']['sentiment'] is None:
                        obj['sentiment'] = 'None'
                    else:
                        obj['sentiment'] = message['entities']['sentiment'][
                            'basic']
                    csvfile.writerow(obj)
                    file.flush()

            print(f"API HITS: {api_hits} {symbol[:-2]}.")

            # No more messages
            if not response['messages']:
                break
            if reach_start_date is True:
                break

        # Add max argument to get older messages
        stocktwit_url = (base_url + symbol + ".json?" + access_token[token]
                         + "max=" + str(last_message_id))
        token = (token + 1) % (len(access_token))

    print(f"Finished {symbol}!")
    print("------------")
    file.close()
