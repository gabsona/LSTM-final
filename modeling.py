import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

import keras_tuner as kt
import os
import joblib
from datetime import datetime
from keras.callbacks import LambdaCallback

from keras_tuner.tuners import RandomSearch, BayesianOptimization
from tensorflow.keras import initializers



parameters = {'batch_size': [32 ,64 ,128],
              'epochs': [50],
              'optimizer__learning_rate': [2, 1, 0.4, 0.2, 1E-1, 1E-3, 1E-5]}

# parameters = {'batch_size': [32 ,64 ,128],
#               'epochs': [50],
#               'optimizer__learning_rate': [2, 1, 0.2, 1E-1, 1E-5]}

# parameters = {'batch_size':[32],
#               'epochs': [10],
#               'optimizer__learning_rate': [1E-5]}
# # # # #               # 'model__': [activation':'relu'}


# def makeLSTM(X_train):
#     inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
#     x = LSTM(25, return_sequences=False)(inputs)
#     x = Dropout(0.1)(x)
#     outputs = Dense(1)(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])
#     model.summary()
#
#     return model

def build_model(X_train, loss, optimizer): #changed the layer of relu
    """
    Args:
        loss: loss function for NN
        optimizer: optimizer for NN

    Returns:
         grid_model: compiled model
    """
    grid_model = Sequential()
    # 1st LSTM layer
    initializer = tf.keras.initializers.GlorotUniform()

    grid_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=initializer)) # (30,4)
    grid_model.add(Dropout(0.2)) # 20% of the units will be dropped

    # #2nd LSTM layer
    # grid_model.add(LSTM(50, return_sequences=True))
    # grid_model.add(Dropout(0.2))

    # 3rd LSTM layer
    # grid_model.add(LSTM(units=50, return_sequences=True))
    # grid_model.add(Dropout(0.5))

    # 4th LSTM layer, we wont use return sequence true in last layers as we dont want to previous output
    grid_model.add(LSTM(units=50, kernel_initializer='glorot_uniform'))
    grid_model.add(Dropout(0.5))
    grid_model.add(Dense(25))
    # Output layer , we wont pass any activation as its continous value model
    grid_model.add(Dense(1, activation = 'sigmoid'))
    grid_model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])
    print('grid_model:', grid_model)

    return grid_model


def main_model(grid_model, problem_type):
    """
    Creates the model depending on the problem type
    Args:
      grid_model: already compiled model
      problem_type: classification or regression

    Returns:
        model: created model
    """

    if problem_type == 'regression':
        model = KerasRegressor(build_fn=grid_model, verbose=1) #, callbacks=[print_weights1, print_weights2])
    if problem_type == 'classification':
        model = KerasClassifier(build_fn=grid_model, verbose=1) #, callbacks=[print_weights1, print_weights2])

    return model

def best_model(X_train, y_train,X_test,y_test,  model, cv):
    """
    Implements cross-validation with given parameters

    Args:
        X_train, y_train: data on which model will be fitted
        cv: cross-validation size

    Returns:
        my_model: model with best parameters
        grid_result: fitted gridsearchcv
    """
    grid_search = GridSearchCV(estimator = model, param_grid = parameters, cv = 5, return_train_score=True, scoring='neg_mean_squared_error')

    # with tf.device('/gpu:0'):
    #     model.fit(X_train, y_train)
    grid_result = grid_search.fit(X_train, y_train,validation_data=(X_test,y_test))
    print('grid search', grid_result)
    print('grid search results', grid_result.cv_results_)


    my_model = grid_result.best_estimator_
    print('my model: ', my_model)

    # saving the model
    # cwd = os.getcwd()
    # dir = os.path.join(cwd, 'saved_models_' + datetime.today().strftime('%d.%m'))
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # joblib.dump(my_model, dir + f'\\model_{ticker}.pkl')

    print('Keys: ', my_model.history_.keys())
    return my_model, grid_result

# def model_fit(X_train, y_train, X_test, y_test, model):
#     """
#     Function needed in case model trained without cross validation
#
#     """
#
#     model.fit(X_train, y_train, epochs=50,validation_data=(X_test,y_test), validation_split=0.2, batch_size=64)
#
#     return model


def model_building(X_train, y_train, X_test, y_test):
    initializer = tf.keras.initializers.GlorotUniform()

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=initializer))

    model.add(Dropout(0.1))
    model.add(LSTM(units=50))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

    print('model:', model)

    return model, history

