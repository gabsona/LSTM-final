from data_engineering import *
from feature_engineering import *
from modeling import *
from prediction import *
from visualisation import *
from helper_functions import *
from bayesian import *

import os
from csv import DictWriter, writer
from datetime import datetime
import tensorflow as tf
import os.path
import csv


def final_pred(ticker, start_date = '2018-01-01', end_date ='2022-01-01', interval = '1d',  change='only close change', division='by date',split_criteria='2021-01-01', scale='yes', step_size=30, loss = 'mse', problem_type = 'regression'):

    data = download_data(ticker, start_date, end_date, interval)
    # data = all_indicators(data) # adds TIs
    data_tr = data_transform(data, change = change)
    # data_tr = data['Close'] #taking only close values

    # target_col_name = data_tr.columns[-2] # changes from -1
    # print('Target column: ', target_col_name)

    target_col_name = 'Close_abs_change'

    X_train, y_train, X_test, y_test, scaler = data_split(data_tr, division, split_criteria, scale, step_size, target_col_name)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    #gridsearch

    # grid_model = build_model(X_train, loss, optimizer='adam')
    # model = main_model(grid_model, problem_type)

    # my_model, grid_result = best_model(X_train, y_train,X_test,y_test, model, cv=5)

    my_model, history = model_building(X_train, y_train, X_test, y_test, 50, 'mse', 'adam', 300)
    # my_model, history = bidirectional_lstm_model(X_train, y_train, X_test, y_test)
    # grid_mean_train = grid_result.cv_results_['mean_train_score']
    # grid_mean_test = grid_result.cv_results_['mean_test_score']


    # #CuDNNLSTM
    # model = makeLSTM(X_train)
    # my_model = model_fit(X_train, y_train, model)

    # #bayesian
    # # model = keras_tuner(hp, X_train, y_train)
    # tuner = BayesianOptimization(keras_tuner, objective='mse', max_trials=30, executions_per_trial=1)
    # tuner.search(x=X_train, y=y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), )
    # best_model = tuner.get_best_models(num_models=1)[0]
    # y_pred = best_model.predict(X_test[0].reshape((1, X_test[0].shape[0], X_test[0].shape[1])))
    # print(y_pred)

    # creating output train test arrays for using them as an input for prediction function
    y_train_close = np.array(data_tr.loc[start_date:'2021-01-01', target_col_name][step_size:])  # 'Close_abs_change' added
    y_test_close = np.array(data_tr.loc['2021-01-01':, target_col_name][step_size:]) #'Close_abs_change' added

    preds_train, score_train = prediction(my_model, y_train_close, X_train, scaler, loss,data_tr, target_col_name) #y_test_close_change
    preds_test, score_test = prediction(my_model, y_test_close, X_test, scaler, loss,data_tr, target_col_name)

    d_train = {'Close_actual': y_train_close, 'Close_prediction': preds_train} #y_test_close_change
    d_test = {'Close_actual': y_test_close, 'Close_prediction': preds_test} #y_test_close_change
    # d = {'Close_actual_change': y_train_close_change, 'Close_prediction_change': preds}

    data_pred_train = pd.DataFrame(data=d_train, index=data[step_size:len(preds_train) + step_size].index) #data[-len(preds):].index
    data_pred_test = pd.DataFrame(data=d_test, index=data[-len(preds_test):].index)
    print('data_pred_train',data_pred_train.head())

    df_preds_train, precision_train, recall_train, f1_train, clf_acc_train = classification(data_pred_train, data, target_col_name, step_size, df_type_='train', change=change)
    df_preds_test, precision_test, recall_test, f1_test, clf_acc_test = classification(data_pred_test, data, target_col_name, step_size, df_type_='test', change=change)
    # print('df_preds_train', df_preds_train)
    # print('df_preds_test', df_preds_test)
    df_preds_abs_train = upd_df(df_preds_train, change = change)
    df_preds_abs_test = upd_df(df_preds_test, change = change)
    print('df_preds_abs', df_preds_abs_test)
    # summarize results
    # print("Best Mean cross-validated training accuracy score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']

    plot_results(ticker, df_preds_train, change=change, df_type='train')
    plot_results(ticker, df_preds_test, change=change, df_type='test')
    # plot_train_val(grid_result, ticker)
    plot_loss(history, my_model, ticker)
    # best_score = grid_result.best_score_
    # best_params = grid_result.best_params_

    min_loss_train = (min(history.history['loss']))
    min_loss_val = (min(history.history['val_loss']))
    final_loss_train = history.history['loss'][-1]
    final_loss_val = history.history['val_loss'][-1]

    return score_train, score_test, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, min_loss_train, min_loss_val, final_loss_train, final_loss_val

    # return best_score, score_train, score_test, best_params, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, grid_mean_train, grid_mean_test


    # return df_preds, df_preds_abs, classification_accuracy, precision, recall, f1, acc, best_score, best_params, mse_train, mse_test

def makemydir(df, stock, folder_name, df_type):
    cwd = os.getcwd()
    dir = os.path.join(cwd, folder_name + datetime.today().strftime('%d.%m'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    # os.chdir(dir)
    df.to_csv(dir + f'\\df_{stock}_{df_type}.csv')



stocks = ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'NVDA', 'JNJ', 'UNH', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']

# stocks = ['GOOG', 'HD', 'LLY', 'MSFT', 'NFLX', 'NVDA', 'PFE', 'TSLA', 'TWTR', 'UNH'] #stocks that had underfitting problem
# stocks = ['V', 'MSFT'] # no PG


for stock in stocks:
    print('stock: ', stock)
    # best_score, score_train, score_test, best_params, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, acc_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, acc_test = final_pred(stock)
    score_train, score_test, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, min_loss_train, min_loss_val, final_loss_train, final_loss_val = final_pred(stock) #,unit_num, loss, optimizer, epoch)

    # df_preds, df_preds_abs, clf_acc,precision, recall, f1, acc,  score, best_params, mse_train, mse_test = final_pred(stock, change='absolute', df_type='train')
    # makemydir(df_preds_train, stock, "Stock Price Prediction (absolute change) ", df_type = 'train')
    makemydir(df_preds_abs_train, stock, "Only close change 300ep mse loss 30 days ", df_type = 'train')
    # makemydir(df_preds_test, stock, "Stock Price Prediction (absolute change) ", df_type = 'train')
    makemydir(df_preds_abs_test, stock, "Only close chang 300ep mse loss 30 days ", df_type = 'test')
    # dict_append = {'Stock': stock, 'Accuracy_train':clf_acc_train, 'Accuracy_test':clf_acc_test,'Precision_train': precision_train, 'Precision_test': precision_test,'Recall_train': recall_train, 'Recall_test': recall_test,'F1_train': f1_train,'F1_test': f1_test, 'Acc_train': acc_train, 'Acc_test': acc_test,'Best_score': best_score, 'Score_train':score_train, 'Score_test':score_test,'Best Parameters':best_params}

    # Open your CSV file in append mode
    # Create a file object for this file
    cwd_main = os.getcwd()
    dir_dict = os.path.join(cwd_main, 'Scores')
    if not os.path.exists(dir_dict):
        os.makedirs(dir_dict)
    os.chdir(dir_dict)
    field_names = ['Stock', 'Accuracy_train', 'Accuracy_test', 'Precision_train', 'Precision_test', 'Recall_train', 'Recall_test', 'F1_train', 'F1_test', 'Score_train', 'Score_test', 'Min loss train', 'Min loss val', 'Final loss train', 'Final loss val']
    dict_append = {'Stock': stock, 'Accuracy_train': clf_acc_train, 'Accuracy_test':clf_acc_test,'Precision_train': precision_train, 'Precision_test': precision_test,'Recall_train': recall_train, 'Recall_test': recall_test,'F1_train': f1_train,'F1_test': f1_test, 'Score_train':score_train, 'Score_test':score_test, 'Min loss train': min_loss_train, 'Min loss val': min_loss_val, 'Final loss train': final_loss_train, 'Final loss val': final_loss_val}

    file_description = 'only_close_change_300ep_mseloss_30days'
    file_name = 'dict_'+ datetime.today().strftime('%d.%m')+'_'+ file_description +'.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)

        if not file_exists:
            dictwriter_object.writeheader()  # file doesn't exist yet, write a header

        # Passing the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_append)

        # Closing the file object
        f_object.close()
    os.chdir(cwd_main)

def stats():
    df = pd.read_csv(dir_dict +'\\'+ file_name)
    print('df', df)
    print(df.mean(axis=0).values)
    list_values = df.mean(axis=0).values.tolist()
    list_values.insert(0,'Average values')
    list_values_for_comparing = df.mean(axis=0).values.tolist()
    list_values_for_comparing.append(file_description)
    print('list_values_for_comparing',list_values_for_comparing)
    means = pd.DataFrame(df.mean(axis=0)).T
    print(df.append(means))
    print(df.mean(axis=0))
    with open(dir_dict +'\\'+ file_name,"a") as f_object:
        # dictwriter_object = DictWriter(filee)
        writer_object = writer(f_object)
        writer_object.writerow(list_values)
        f_object.close()
    # with open(dir_dict+'\\'+ 'comparing_models.csv', "w+") as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(['Accuracy_train', 'Accuracy_test', 'Precision_train', 'Precision_test', 'Recall_train', 'Recall_test', 'F1_train', 'F1_test', 'Score_train', 'Score_test', 'Min loss train', 'Min loss val', 'Final loss train', 'Final loss val'])
    #     f.close()
    with open(dir_dict+'\\'+ 'comparing_models.csv', "a") as obj:

        writer_object = writer(obj)
        writer_object.writerow(list_values_for_comparing)
        obj.close()

stats()