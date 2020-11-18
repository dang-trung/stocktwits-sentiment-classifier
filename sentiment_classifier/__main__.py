#!/usr/bin/env python3
"""Main module.

This runs when ...
"""
import pandas as pd
from sentiment_classifier.get_stocktwits_message import get_stocktwits_message

if __name__ == '__main__':
    st_cashtag = pd.read_csv('data//symbols.csv', header=None)
    symbols = st_cashtag[1]
    symbols = symbols[symbols.str.endswith('.X')]  # find crypto symbols
    symbols = symbols.to_list()  #
    start = "2014-11-28"
    end = "2020-07-26"
    for symbol in symbols:
        get_stocktwits_message(symbol=symbol, start=start, end=end,
                               file_name=f"data//stocktwits.csv")
