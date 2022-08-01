from feature_engineering import *
from data_engineering import *
from modeling import *
from prediction import *
from visualisation import *
from helper_functions import *

import os
from csv import DictWriter
from datetime import datetime
import tensorflow as tf

def final_pred(ticker, change='absolute', df_type='test'):
    data = download_data(ticker, '2018-01-01', '2022-01-01', '1d')
    # data = all_indicators(data) # adds TIs
    data.dropna(inplace = True)
    data_tr = data_transform(data, change)
    X_train, y_train, X_test, y_test, scaler = data_split(data_tr.iloc[:, :-1], division='by date',
                                                          split_criteria='2021-01-01', scale='yes', step_size=30)
    grid_model = build_model(X_train, loss='mse', optimizer='adam')
    # grid_model = KerasRegressor(build_fn=grid_model, verbose=1)
    model = reg_model(grid_model)
    my_model, grid_result = best_model(X_train, y_train, model, cv=3)
    # dataset_test = data.iloc[:, :-1].loc['2021-01-01':]
    # y_test_change = data_tr.loc['2021-01-01':]
    # y_test_change = np.array(y_test_change.iloc[30:,3])

    #y_test_close = np.array(data.loc['2021-01-01':, 'Close'][30:])
    y_test_close_change = np.array(data_tr.loc['2021-01-01':, 'Close_abs_change'][30:])
    y_train_close_change = np.array(data_tr.loc['2018-01-01':'2021-01-01', 'Close_abs_change'][30:])
    preds, score = prediction(my_model, y_train_close_change, X_train, scaler, loss='mse') #y_test_close_change
    print('preds:',preds.shape)
    print(y_train_close_change)
    d = {'Close_actual_change': y_train_close_change, 'Close_prediction_change': preds}
    # d = {'Close_actual_change': y_train_close_change, 'Close_prediction_change': preds}
    data_pred = pd.DataFrame(data=d, index=data[30:len(preds)+30].index) #data[-len(preds):].index
    print('data_pred', data_pred.head())
    print('data_pred', data_pred.tail())
    df_preds, classification_accuracy, precision, recall, f1, acc = classification(data_pred, data, change=change)
    print(df_preds)
    df_preds_abs = upd_df(df_preds)

    # summarize results
    print("Best Mean cross-validated training accuracy score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    mse_train = mean_squared_error(y_train, grid_result.predict(X_train))
    mse_test = mean_squared_error(y_test, grid_result.predict(X_test))
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Train MSE : {}".format(mean_squared_error(y_train, grid_result.predict(X_train))))
    print("Test  MSE : {}".format(mean_squared_error(y_test, grid_result.predict(X_test))))

    print("\nTrain R^2 : {}".format(grid_result.score(X_train, y_train)))
    print("Test  R^2 : {}".format(grid_result.score(X_test, y_test)))

    plot_results(ticker, df_preds_abs, change=change, df_type = df_type)
    plot_loss(my_model, ticker, df_type = df_type)
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_

    return df_preds, df_preds_abs, classification_accuracy, precision, recall, f1, acc, best_score, best_params, mse_train, mse_test

def makemydir(df, stock, folder_name, df_type = 'test'):
    cwd = os.getcwd()
    dir = os.path.join(cwd, folder_name + datetime.today().strftime('%d.%m'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    # os.chdir(dir)
    df.to_csv(dir + f'\\df_{stock}_{df_type}_change.csv')


acc_list = [] # add training accuracy
precision = []
recall = []
f1 = []
acc = []
scores = []
best_params_ = []
mse_train_ = []
mse_test_ = []
dict_acc = {'Stock': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'Acc': [], 'Score': [], 'MSE train': [], 'MSE test': [], 'Best Parameters': []}
df_acc = pd.DataFrame(dict_acc)
df_acc.to_csv('dict_'+ datetime.today().strftime('%d.%m')+'.csv', index = False)

stocks = ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'FB', 'NVDA', 'JNJ', 'UNH', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']
# stocks = ['JNJ', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO']
# stocks = ['CVX', 'MA', 'WMT', 'HD']
# stocks = ['NFLX'] # no PG

for stock in stocks:
    df_preds, df_preds_abs, clf_acc,precision, recall, f1, acc,  score, best_params, mse_train, mse_test = final_pred(stock, change='absolute', df_type='train')
    makemydir(df_preds, stock, "Stock Price Prediction (absolute change) ", df_type = 'train')
    makemydir(df_preds_abs, stock, "Stock Price Prediction(with added changes) (absolute change) ", df_type = 'train')
    dict_append = {'Stock': stock, 'Accuracy':clf_acc, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Acc': acc, 'Score': score, 'MSE train': mse_train, 'MSE test':mse_test, 'Best Parameters':best_params}
    # Open your CSV file in append mode
    # Create a file object for this file
    with open('dict_'+ datetime.today().strftime('%d.%m')+'.csv', 'a', newline='') as f_object:

        fieldnames = ['Stock', 'Accuracy', 'Precision', 'Recall', 'F1', 'Acc', 'Score', 'MSE train', 'MSE test', 'Best Parameters']
        dictwriter_object = DictWriter(f_object, fieldnames = dict_append)

        # Passing the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_append)

        # Closing the file object
        f_object.close()