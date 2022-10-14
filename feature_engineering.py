import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#add scaler type input
def data_split(data, division, split_criteria, scale, step_size, target_col_name):
    """
    divides data into train and test parts
    Args:
         division: division type, either by date or by legnth percentage
         split_criteria: date for division by date, or percentage number for percentage division
         scale: deciding whether data will be scaled or not
         step_size: length of training part (i.e. step_size=30 means we take 30 days stock prices to predict 31st day price)
    Returns:
         X_train, y_train, X_test, y_test
         scaler: scaler will be needed for inverse scaling of predictions
    """
    if division == 'by date':
        dataset_train = data.loc[:split_criteria]
        dataset_test = data.loc[split_criteria:]

    elif division == 'by percentage':
        dataset_train = data.iloc[:int(len(data) * split_criteria), :]
        dataset_test = data.iloc[int(len(data) * split_criteria):, :]

    if scale == 'yes':
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_for_training = scaler.fit_transform(dataset_train) #removed values.reshape(-1, 1) for minmaxscaler
        df_for_test = scaler.transform(dataset_test)

    else:
        scaler = None
        df_for_training = dataset_train
        df_for_test = dataset_test

    df_for_training = np.array(df_for_training)
    print('df_for_training', df_for_training)
    df_for_test = np.array(df_for_test)

    dataX = []
    dataY = []
    for i in range(step_size, len(df_for_training)):
        dataX.append(df_for_training[i - step_size:i, 0:df_for_training.shape[1]]) #'-1' added for classification
        dataY.append(df_for_training[i, data.columns.get_loc(target_col_name)]) #3
        X_train, y_train = np.array(dataX), np.array(dataY)

    dataX = []
    dataY = []
    for i in range(step_size, len(df_for_test)):
        dataX.append(df_for_test[i - step_size:i, 0:df_for_test.shape[1]]) #'-1' added for classification
        dataY.append(df_for_test[i, data.columns.get_loc(target_col_name)]) #3
        X_test, y_test = np.array(dataX), np.array(dataY)
    # print("X train y train")
    # print(X_train[0])
    # print(y_train[0])
    # print(X_train[1])
    # print(y_train[1])

    # print(X_test[0])
    # print(y_test[0])
    # print(X_test[1])
    # print(y_test[1])

    return X_train, y_train, X_test, y_test, scaler