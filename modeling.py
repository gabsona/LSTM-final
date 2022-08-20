import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import keras_tuner as kt
import os
import joblib
from datetime import datetime
from keras.callbacks import LambdaCallback

from keras_tuner.tuners import RandomSearch, BayesianOptimization
from tensorflow.keras import initializers


# parameters = {'batch_size': [32 ,64 ,128],
#               # 'batch_size': [32],
#               'epochs': [50],
#               'optimizer__learning_rate': [2, 1, 0.4, 0.2, 1E-1, 1E-3, 1E-5]}

parameters = {'batch_size': [32],
              'epochs': [30],
              'optimizer__learning_rate': [0.001]}
# #               # 'model__activation':'relu'}

# parameters = {'batch_size': [16 ,32]}



def build_model(X_train, loss, optimizer): #changed the layer of relu

    grid_model = Sequential()
    # 1st LSTM layer
    initializer = tf.keras.initializers.GlorotUniform()

    grid_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=initializer)) # (30,4)
    grid_model.add(Dropout(0.2)) # 20% of the units will be dropped

    # 2nd LSTM layer
    # grid_model.add(LSTM(50, return_sequences=True))
    # grid_model.add(Dropout(0.2))

    # 3rd LSTM layer
    # grid_model.add(LSTM(units=50, return_sequences=True))
    # grid_model.add(Dropout(0.5))

    # 4th LSTM layer, we wont use return sequence true in last layers as we dont want to previous output
    grid_model.add(LSTM(units=50, kernel_initializer='glorot_uniform'))
    grid_model.add(Dropout(0.5))

    # Output layer , we wont pass any activation as its continous value model
    grid_model.add(Dense(1))

    grid_model.compile(loss = loss,optimizer = optimizer,metrics=['accuracy'])
    print('grid_model:', grid_model)

    return grid_model


def main_model(grid_model, problem_type):
    # for layer in model.layers:
    #     weights = layer.get_weights()
    # for layer in model.layers: print(layer.get_config(), layer.get_weights())
    print('grid model layers', grid_model.layers)
    # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print('WEIGHTS 1:', grid_model.layers[0].get_weights()))
    # print_weights = LambdaCallback(on_batch_end=lambda batch, logs: (for layer in grid_model.layers print('WEIGHTS:',layer.get_weights())))
    layer1_weights = []
    layer2_weights = []
    # print_weights1 = LambdaCallback(on_batch_end=lambda batch, logs: print('WEIGHTS 1:', grid_model.layers[0].get_weights()[0], grid_model.layers[0].get_weights()[0].shape))
    # print_weights2 = LambdaCallback(on_batch_end=lambda batch, logs: print('WEIGHTS 2:', grid_model.layers[2].get_weights()[0], grid_model.layers[0].get_weights()[0].shape))

    # print_weights1 = LambdaCallback(on_batch_end=lambda batch, logs: pd.DataFrame(layer1_weights.append(grid_model.layers[0].get_weights()[0])).to_csv(f'lw1_{ticker}.csv'))
    # print_weights2 = LambdaCallback(on_batch_end=lambda batch, logs: layer2_weights.append(grid_model.layers[1].get_weights()[0]))

    if problem_type == 'regression':
        model = KerasRegressor(build_fn=grid_model, verbose=1) #, callbacks=[print_weights1, print_weights2])
    if problem_type == 'classification':
        model = KerasClassifier(build_fn=grid_model, verbose=1) #, callbacks=[print_weights1, print_weights2])

    # lw1_df = pd.DataFrame(layer1_weights)
    # lw2_df = pd.DataFrame(layer2_weights)
    # lw1_df.to_csv(f'lw1_{ticker}.csv')
    # lw2_df.to_csv(f'lw2_{ticker}.csv')
    return model

def best_model(X_train, y_train, model, cv):
    grid_search = GridSearchCV(estimator = model, param_grid = parameters, cv = cv)

    # with tf.device('/gpu:0'):
    #     model.fit(X_train, y_train)
    grid_result = grid_search.fit(X_train, y_train)
    my_model = grid_result.best_estimator_
    # weight = my_model.build_fn.get_weights()
    # np.savetxt('weight.csv', weight, fmt='%s', delimiter=',')

    # print('params:', grid_result.best_params_)
    # print('grid_result:', grid_result)
    # print('my_model:', my_model)
    # to_be_saved_model = my_model.fit(X_train, y_train, callbacks=None)
    # print('to_be_saved_model',to_be_saved_model)

    # saving the model
    # cwd = os.getcwd()
    # dir = os.path.join(cwd, 'saved_models_' + datetime.today().strftime('%d.%m'))
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # joblib.dump(my_model, dir + f'\\model_{ticker}.pkl')

    print('Keys: ', my_model.history_.keys())
    return my_model, grid_result