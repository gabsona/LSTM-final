import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parameters = {'batch_size': [32 ,64 ,128],
              'epochs': [100],
              'optimizer__learning_rate': [0.4, 0.2, 1E-0, 1E-1, 1E-3, 1E-5, 1E-7]}

# parameters = {'batch_size': [16 ,32]}

def build_model(X_train, loss = 'mse', optimizer = 'adam'):

    grid_model = Sequential()
    # 1st LSTM layer
    grid_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # (30,4)
    # grid_model.add(Dropout(0.2)) # 20% of the units will be dropped
    # 2nd LSTM layer
    # grid_model.add(LSTM(50, return_sequences=True))
    # grid_model.add(Dropout(0.2))
    # 3rd LSTM layer
    # grid_model.add(LSTM(units=50, return_sequences=True))
    # grid_model.add(Dropout(0.5))
    # 4th LSTM layer
    grid_model.add(LSTM(units=50))
    # grid_model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    grid_model.add(Dense(1))
    grid_model.compile(loss = loss,optimizer = optimizer)

    return grid_model

def reg_model(grid_model):
    model = KerasRegressor(build_fn=grid_model, verbose=1)
    return model

def best_model(X_train, y_train, grid_model, cv = 3):
    grid_search  = RandomizedSearchCV(grid_model, parameters, cv = cv)

    # with tf.device('/gpu:0'):
    #     model.fit(X_train, y_train)
    grid_result = grid_search.fit(X_train, y_train)
    my_model = grid_result.best_estimator_
    return my_model, grid_result




