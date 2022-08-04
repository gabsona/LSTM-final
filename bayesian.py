
import pandas as pd
import math
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
import warnings
warnings.simplefilter("ignore", UserWarning)

def keras_tuner():
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32),return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(y_train.shape[1], activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

# tuner= BayesianOptimization(
#         build_model,
#         seed=random_seed,
#         objective='mse',
#         max_trials=30,
#         executions_per_trial=1
# #         )
# # tuner_mlp = kt.tuners.BayesianOptimization(
# #     model,
# #     seed=random_seed,
# #     objective='val_loss',
# #     max_trials=30,
# #     directory='.',
# #     project_name='tuning-mlp')
# tuner.search(
#         x=X_train,
#         y=y_train,
#         epochs=50,
#         batch_size=128,
#         validation_data=(X_test,y_test),)
# best_model = tuner.get_best_models(num_models=1)[0]
#
# y_pred=best_model.predict(X_test[0].reshape((1,X_test[0].shape[0], X_test[0].shape[1])))
