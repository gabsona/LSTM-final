from feature_engineering import *
from data_engineering import *
from modeling import *
from prediction import *
from visualisation import *
from helper_functions import *
from bayesian import *

import os
from csv import DictWriter
from datetime import datetime
import tensorflow as tf

def final_pred(ticker, start_date = '2018-01-01', end_date ='2022-01-01', interval = '1d',  change='for classification', division='by date',split_criteria='2021-01-01', scale='yes', step_size=30, loss='binary_crossentropy' , problem_type = 'classification'):
    data = download_data(ticker, start_date, end_date, interval)
    # data = all_indicators(data) # adds TIs
    data_tr = data_transform(data, change = change)
    # data_tr = data['Close'] #taking only close values

    target_col_name = data_tr.columns[-1]
    print('Target column: ', target_col_name)

    X_train, y_train, X_test, y_test, scaler = data_split(data_tr, division, split_criteria, scale, step_size, target_col_name)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    #gridsearch
    grid_model = build_model(X_train, loss= loss, optimizer='adam')
    model = main_model(grid_model, problem_type)
    # print('layer1_weights', layer1_weights)
    # lw1_df = pd.DataFrame(layer1_weights)
    # lw2_df = pd.DataFrame(layer2_weights)
    # lw1_df.to_csv(f'lw1_{ticker}.csv')
    # lw2_df.to_csv(f'lw2_{ticker}.csv')
    my_model, grid_result = best_model(X_train, y_train, model, cv=3)

    # #bayesian
    # # model = keras_tuner(hp, X_train, y_train)
    # tuner = BayesianOptimization(keras_tuner, objective='mse', max_trials=30, executions_per_trial=1)
    # tuner.search(x=X_train, y=y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), )
    # best_model = tuner.get_best_models(num_models=1)[0]
    # y_pred = best_model.predict(X_test[0].reshape((1, X_test[0].shape[0], X_test[0].shape[1])))
    # print(y_pred)

    print('data_tr: ', data_tr.head())
    y_train_close = np.array(data_tr.loc['2018-01-01':'2021-01-01',target_col_name][step_size:])  # 'Close_abs_change' added
    y_test_close = np.array(data_tr.loc['2021-01-01':, target_col_name][step_size:]) #'Close_abs_change' added

    preds_train, score_train = prediction(my_model, y_train_close, X_train, scaler, loss) #y_test_close_change
    preds_test, score_test = prediction(my_model, y_test_close, X_test, scaler, loss)

    d_train = {'Close_actual': y_train_close, 'Close_prediction': preds_train} #y_test_close_change
    d_test = {'Close_actual': y_test_close, 'Close_prediction': preds_test} #y_test_close_change
    # d = {'Close_actual_change': y_train_close_change, 'Close_prediction_change': preds}
    data_pred_train = pd.DataFrame(data=d_train, index=data[step_size:len(preds_train) + step_size].index) #data[-len(preds):].index
    data_pred_test = pd.DataFrame(data=d_test, index=data[-len(preds_test):].index)
    # print('data_pred_train:',data_pred_train)
    # print('data_pred_test:', data_pred_test)
    # print('data_pred', data_pred.head())
    # print('data_pred', data_pred.tail())
    df_preds_train, clf_acc_train, precision_train, recall_train, f1_train, acc_train = classification(data_pred_train, data,df_type_='train', change='no_change')
    df_preds_test, clf_acc_test, precision_test, recall_test, f1_test, acc_test = classification(data_pred_test, data, df_type_='test', change='no_change')
    print('df_preds_train', df_preds_train)
    df_preds_abs_train = upd_df(df_preds_train, change = change)
    df_preds_abs_test = upd_df(df_preds_test, change = change)

    # summarize results
    print("Best Mean cross-validated training accuracy score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    plot_results(ticker, df_preds_abs_train, change=change, df_type='train')
    plot_results(ticker, df_preds_abs_test, change=change, df_type='test')
    plot_loss(my_model, ticker)
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_

    return best_score, score_train, score_test, best_params, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, acc_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, acc_test

    # return df_preds, df_preds_abs, classification_accuracy, precision, recall, f1, acc, best_score, best_params, mse_train, mse_test

def makemydir(df, stock, folder_name, df_type = 'test'):
    cwd = os.getcwd()
    dir = os.path.join(cwd, folder_name + datetime.today().strftime('%d.%m'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    # os.chdir(dir)
    df.to_csv(dir + f'\\df_{stock}_{df_type}_close_OHL.csv')


acc_list = [] # add training accuracy
precision = []
recall = []
f1 = []
acc = []
scores = []
best_params_ = []
mse_train_ = []
mse_test_ = []
dict_acc = {'Stock': [], 'Accuracy_train': [], 'Accuracy_test': [],'Precision_train': [], 'Precision_test': [],'Recall_train': [],'Recall_test': [], 'F1_train': [], 'F1_test': [], 'Acc_train': [], 'Acc_test': [],'Best_score': [], 'Score_train':[], 'Score_test':[],'Best_parameters': []}
df_acc = pd.DataFrame(dict_acc)
# df_acc.to_csv('dict_'+ datetime.today().strftime('%d.%m')+'.csv', index = False)

# stocks = ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'NVDA', 'JNJ', 'UNH', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']
# stocks = ['JNJ', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO']
# stocks = ['CVX', 'MA']
stocks = ['NFLX'] # no PG

for stock in stocks:
    print('stock: ', stock)
    best_score, score_train, score_test, best_params, df_preds_train, df_preds_abs_train, clf_acc_train, precision_train, recall_train, f1_train, acc_train, df_preds_test, df_preds_abs_test, clf_acc_test, precision_test, recall_test, f1_test, acc_test = final_pred(stock)
    # df_preds, df_preds_abs, clf_acc,precision, recall, f1, acc,  score, best_params, mse_train, mse_test = final_pred(stock, change='absolute', df_type='train')
    # makemydir(df_preds_train, stock, "Stock Price Prediction (absolute change) ", df_type = 'train')
    makemydir(df_preds_abs_train, stock, "Close unchanged", df_type = 'train')
    # makemydir(df_preds_test, stock, "Stock Price Prediction (absolute change) ", df_type = 'train')
    makemydir(df_preds_abs_test, stock, "Close unchanged", df_type = 'test')
    dict_append = {'Stock': stock, 'Accuracy_train':clf_acc_train, 'Accuracy_test':clf_acc_test,'Precision_train': precision_train, 'Precision_test': precision_test,'Recall_train': recall_train, 'Recall_test': recall_test,'F1_train': f1_train,'F1_test': f1_test, 'Acc_train': acc_train, 'Acc_test': acc_test,'Best_score': best_score, 'Score_train':score_train, 'Score_test':score_test,'Best Parameters':best_params}
    # Open your CSV file in append mode
    # Create a file object for this file
    with open('dict_close_'+ datetime.today().strftime('%d.%m')+'.csv', 'a', newline='') as f_object:

        # fieldnames = ['Stock', 'Accuracy', 'Precision', 'Recall', 'F1', 'Acc', 'Score', 'MSE train', 'MSE test', 'Best Parameters']
        fieldnames = ['Stock', 'Accuracy_train', 'Accuracy_test','Precision_train', 'Precision_test','Recall_train','Recall_test', 'F1_train', 'F1_test', 'Acc_train', 'Acc_test','Best_score', 'Score_train', 'Score_test','Best_parameters']
        dictwriter_object = DictWriter(f_object, fieldnames = dict_append)

        # Passing the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_append)

        # Closing the file object
        f_object.close()