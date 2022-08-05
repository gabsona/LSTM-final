import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



def prediction(model, original, X_test, scaler, loss = 'mse'):
  print('X_test', X_test)
  print('X_test', X_test.shape)
  prediction = model.predict(X_test)
  # print('pred1 ', prediction)
  prediction = prediction.reshape(prediction.shape[0],1)
  # print('pred2 ', prediction)
  pred = scaler.inverse_transform(prediction)
  print('pred3 ', pred)
  pred = np.reshape(pred, (len(prediction),))# X_test.shape[2]))
  print(original.shape)
  print(pred.shape)
  # print('pred4 ', pred)
  # prediction_copies_array = np.repeat(prediction, X_test.shape[2], axis=-1) #change this one
  # print('pred3 ', prediction_copies_array)
  # pred = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction), X_test.shape[2])))[:,3]
  #pred = np.reshape(pred, (len(prediction), X_test.shape[2]))[:, 3]
  if loss == 'mse':
    testScore = mean_squared_error(original, pred)
  if loss == 'mape':
    testScore = mean_absolute_percentage_error(original, pred)

  return pred, testScore



def classification(data, data_main, df_type_, change):
    if change == 'no change':

        data['Actual_change'] = np.where(data['Close_actual_change'] < data['Close_actual_change'].shift(1), 0, 1)
        data['Pred_change'] = np.where(data['Close_actual_change'] > data['Close_prediction_change'].shift(-1), 0, 1)
        data['Pred_change'] = data.Pred_change.shift(1)
        data = data[1:]
        data['Pred_change'] = data['Pred_change'].astype(int)
        classification_accuracy = len(data[(data.Actual_change == data.Pred_change)]) / len(data)

    elif change == 'absolute':

        data['Actual_change'] = np.where(data['Close_actual_change'] < 0, 0, 1)
        data['Pred_change'] = np.where(data['Close_prediction_change'] < 0, 0, 1)
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

def upd_df(df):
    #df = pd.read_csv(f'C:\Stock Price Prediction\df_{ticker}.csv')
    Added_changes = []
    for i in range(len(df)):
      Added_changes.append(df.Close_actual[0] + df.Close_prediction_change[1] + df.Close_prediction_change[1:i].sum()) # changed from Close_actual

    df['Added_changes'] = Added_changes
    df['Added_changes'] = df['Added_changes'].shift(-1)
    return df

# def plot_results(ticker, df, data, data_tr, pred, change):
#     plt.figure(figsize=(12, 6))
#     y_test_change = data_tr.loc['2021-01-01':]
#     y_test_change = np.array(y_test_change.iloc[30:, 3]) #takes only Close value
#     plt.plot(y_test_change, color='green', label='Real Price')
#     plt.plot(pred, color='purple', label='Predicted Price')
#     plt.title(f'{ticker}')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()
#
#     if change == 'absolute':
#
#         pd.concat([data.loc['2021-01-01':]['Close'], pd.DataFrame(data.loc['2021-01-01':]['Close'].shift(1) + df.Close_prediction_change)], axis=1)[31:].plot(figsize=(12, 8))
#         plt.title('Close Absolute Change Prediction')
#
#         pd.concat([data.loc['2021-01-01':]['Close'][31:], pd.DataFrame(data.loc['2021-01-01':]['Close'][31] + df.Close_prediction_change)], axis=1).plot(figsize=(12, 8))
#         plt.title('Close Absolute Change Prediction (only adding changes)')
#
#     else:
#         pass

