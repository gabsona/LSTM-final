import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error, precision_score, recall_score, f1_score

import numpy as np
import itertools

import pandas as pd
from matplotlib import pyplot as plt


def prediction(model, actual_values, X, scaler, loss):
    """
    Predicts unseen data
    Args:
        model: model with best parameters
        X: input datan on which prediction should be done
        loss: loss function which estimated the model while training

    Returns:
        pred: predictions
        testScore: the score which estimated the model
    """

    # print('X', X.shape)
    # print('original', actual_values.shape)

    pred = model.predict(X) #added for clf
    print('original', actual_values)
    print('pred1', pred)
    # for classification
    # pred = np.where(pred > 0, 1, 0)
    # print('pred1.1 ', pred[:5])
    # print('pred1.1', pred.shape)

    prediction_copies_array = np.repeat(pred, X.shape[2], axis=-1)
    pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(pred), X.shape[2])))[:, 3]

    # pred = np.reshape(pred, (len(prediction),))# X_test.shape[2]))
    # print(pred.shape)
    # print('pred4 ', pred)
    # prediction_copies_array = np.repeat(prediction, X.shape[2], axis=-1) #change this one
    # print('pred3 ', prediction_copies_array[:5])

    # pred = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction), X.shape[2])))[:,3]
    # print('pred4', pred[:5], pred.shape)
    # pred = np.reshape(pred, (len(prediction), X_test.shape[2]))[:, 3]
    if loss == 'mse':
        testScore = mean_squared_error(actual_values, pred)
    if loss == 'mape':
        testScore = mean_absolute_percentage_error(actual_values, pred)
    if loss == 'binary_crossentropy':
        testScore = accuracy_score(actual_values, pred)
    print('testScore', testScore)
    return pred, testScore



def classification(data, data_main, df_type_, change):
    """
    This function is used when we do regression. Finds whether model can accurately predict the direction of predictions
    Args:
        data: dataset with predictions and actual values
        data_main: initial data
        change: change type oof input data (if there's any change)

    Returns:
        data: dataset with classification results (0,1)

    """
    if change == 'no change':

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
        if df_type_ == 'train':
            data['Close_actual'] = data_main.loc['2018-01-01':'2021-01-01', 'Close'][30:] #'2021-01-01':
        if df_type_ == 'test':
            data['Close_actual'] = data_main.loc['2021-01-01':, 'Close'][30:]  # '2021-01-01':
        # data['Close_prediction'] = dataset_test['Close'].shift(1) + data.Close_prediction_change

    elif change == 'classification':
        data.rename(columns = {'Close_actual': 'Actual_change', 'Close_prediction':'Pred_change'}, inplace = True)

    precision = precision_score(data.Actual_change, data.Pred_change)
    print('precision', precision)
    recall = recall_score(data.Actual_change, data.Pred_change)
    print('recall', recall)
    f1 = f1_score(data.Actual_change, data.Pred_change)
    acc = accuracy_score(data.Actual_change, data.Pred_change)

    return data, precision, recall, f1, acc

def upd_df(df, change):
    """
    Adds column for regression predictions, close price + absolute change predictions
    Args:
        change: input change type
    Returns:
        df: dataframe with column with added changes
    """
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