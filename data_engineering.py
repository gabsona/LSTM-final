import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd


def download_data(ticker, start_date, end_date, interval = '1d'):

  data = yf.download(ticker, start = start_date, end = end_date, interval = interval)

  return data


def data_transform(data, change):
    if change == 'absolute':

        data['Open_abs_change'] = data.Open.shift(-1) - data.Open #make diff()
        data['Open_abs_change'] = data['Open_abs_change'].shift(1)

        data['High_abs_change'] = data.High.shift(-1) - data.High
        data['High_abs_change'] = data['High_abs_change'].shift(1)

        data['Low_abs_change'] = data.Low.shift(-1) - data.Low
        data['Low_abs_change'] = data['Low_abs_change'].shift(1)

        data['Close_abs_change'] = data.Close.shift(-1) - data.Close
        data['Close_abs_change'] = data['Close_abs_change'].shift(1)
        # data['Open_Close_abs_change'] = data['Close'] - data['Open']

        data = data.iloc[1:, 6:]

    elif change == 'only close change':

        data['Close_abs_change'] = data.Close.shift(-1) - data.Close
        data['Close_abs_change'] = data['Close_abs_change'].shift(1)

        # data = data.drop(columns=['Close', 'Adj Close', 'Volume'], axis=1)
        data = data.Close_abs_change

    elif change == 'OHL only close change':

        data['Close_abs_change'] = data.Close.shift(-1) - data.Close
        data['Close_abs_change'] = data['Close_abs_change'].shift(1)

        data = data.drop(columns=['Close', 'Adj Close', 'Volume'], axis=1)

    elif change == 'percentage':

        data['Open_pct_change'] = data.Open.pct_change()
        data['High_pct_change'] = data.High.pct_change()
        data['Low_pct_change'] = data.Low.pct_change()
        data['Close_pct_change'] = data.Close.pct_change()

        data['Open_Close_change'] = (data['Close'] - data['Open']) / data['Open']

        data = data.iloc[1:, 6:]

    elif change == 'no_change':
        data = data.iloc[:,:4]

    else:
        raise Exception('Wrong input')
    print('Data shape: ', data.shape)
    return data

# data = download_data('NFLX', '2018-01-01', '2022-01-01', '1d')
# # data = all_indicators(data) # adds TIs
# data.dropna(inplace = True)
# data_tr = data_transform(data, change='only close')
# print(data_tr)