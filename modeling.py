import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import keras_tuner as kt
import os
import joblib
from datetime import datetime
from keras.callbacks import LambdaCallback

from keras_tuner.tuners import RandomSearch, BayesianOptimization


# parameters = {'batch_size': [32 ,64 ,128],
#               'epochs': [50],
#               'optimizer__learning_rate': [2, 1, 0.4, 0.2, 1E-1, 1E-3, 1E-5]}

parameters = {'batch_size': [64],
              'epochs': [30],
              'optimizer__learning_rate': [2.5]}
# #               # 'model__activation':'relu'}

# parameters = {'batch_size': [16 ,32]}



def build_model(X_train, loss, optimizer): #changed the layer of relu

    grid_model = Sequential()
    # 1st LSTM layer
    grid_model.add(LSTM(100, activation = 'relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # (30,4)
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
    print('grid_model:', grid_model)
    # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print('WEIGHTS:', grid_model.layers[0].get_weights()))

    grid_model.compile(loss = loss,optimizer = optimizer)

    return grid_model


def reg_model(grid_model):
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print('WEIGHTS:', grid_model.layers[0].get_weights()))

    model = KerasRegressor(build_fn=grid_model, verbose=1, callbacks=[print_weights])

    return model

def best_model(X_train, y_train, model, cv, ticker):
    grid_search = GridSearchCV(model, parameters, cv = cv)

    # with tf.device('/gpu:0'):
    #     model.fit(X_train, y_train)
    grid_result = grid_search.fit(X_train, y_train)
    my_model = grid_result.best_estimator_
    print('params:', grid_result.best_params_)
    print('grid_result:', grid_result)
    print('my_model:', my_model)
    # to_be_saved_model = my_model.fit(X_train, y_train, callbacks=None)
    # print('to_be_saved_model',to_be_saved_model)
    # saving the model
    cwd = os.getcwd()
    dir = os.path.join(cwd, 'saved_models_' + datetime.today().strftime('%d.%m'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    joblib.dump(to_be_saved_model, dir + f'\\model_{ticker}.pkl')

    print('Keys: ', my_model.history_.keys())
    return my_model, grid_result