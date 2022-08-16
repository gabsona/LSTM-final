import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



def prediction(model, original, X, scaler, loss ='mse'):
  print('X', X.shape)
  prediction = model.predict(X)
  # print('pred1 ', prediction[:5])
  prediction = prediction.reshape(prediction.shape[0],1)
  # print('pred2 ', prediction[:5])
  # pred = scaler.inverse_transform(prediction)
  # print('pred3 ', pred)
  # pred = np.reshape(pred, (len(prediction),))# X_test.shape[2]))
  print(original.shape)
  # print(pred.shape)
  # print('pred4 ', pred)
  prediction_copies_array = np.repeat(prediction, X.shape[2], axis=-1) #change this one
  # print('pred3 ', prediction_copies_array[:5])
  # print('pred3.shape ', prediction_copies_array.shape)
  pred = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction), X.shape[2])))[:,3]
  # print('pred4', pred[:5], pred.shape)
  print('original', original.shape)
  # pred = np.reshape(pred, (len(prediction), X_test.shape[2]))[:, 3]
  if loss == 'mse':
    testScore = mean_squared_error(original, pred)
  if loss == 'mape':
    testScore = mean_absolute_percentage_error(original, pred)

  return pred, testScore



def classification(data, data_main, df_type_, change):
    if change == 'no_change':

        data['Actual_change'] = np.where(data['Close_actual'] < data['Close_actual'].shift(1), 0, 1)
        data['Pred_change'] = np.where(data['Close_actual'] > data['Close_prediction'].shift(-1), 0, 1)
        data['Pred_change'] = data.Pred_change.shift(1)
        data = data[1:]
        data['Pred_change'] = data['Pred_change'].astype(int)
        classification_accuracy = len(data[(data.Actual_change == data.Pred_change)]) / len(data)

    elif change == 'absolute':

        data['Actual_change'] = np.where(data['Close_actual'] < 0, 0, 1)
        data['Pred_change'] = np.where(data['Close_prediction'] < 0, 0, 1)
        data = data[1:]
        data['Pred_change'] = data['Pred_change'].astype(int)
        if df_type_=='train':
            data['Close_actual'] = data_main.loc['2018-01-01':'2021-01-01', 'Close'][30:] #'2021-01-01':
        if df_type_=='test':
            data['Close_actual'] = data_main.loc['2021-01-01':, 'Close'][30:]  # '2021-01-01':
        # data['Close_prediction'] = dataset_test['Close'].shift(1) + data.Close_prediction_change
    classification_accuracy = len(data[(data.Actual_change == data.Pred_change)]) / len(data)
    precision = precision_score(data.Actual_change, data.Pred_change)
    recall = recall_score(data.Actual_change, data.Pred_change)
    f1 = f1_score(data.Actual_change, data.Pred_change)
    acc = accuracy_score(data.Actual_change, data.Pred_change)

    return data, classification_accuracy, precision, recall, f1, acc

def upd_df(df, change):
    #df = pd.read_csv(f'C:\Stock Price Prediction\df_{ticker}.csv')
    if change == 'absolute':
        Added_changes = []
        for i in range(len(df)): #Close_prediction_change
          Added_changes.append(df.Close_actual[0] + df.Close_prediction[1] + df.Close_prediction[1:i].sum()) # changed from Close_actual

        df['Added_changes'] = Added_changes
        df['Added_changes'] = df['Added_changes'].shift(-1)
    else:
        pass
    return df
