import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

def plot_results(ticker, df, change, df_type, date = datetime.today().strftime('%d.%m')):
    plt.figure(figsize=(12, 6))
    print(df)
    plt.plot(df.Close_actual, color='green', label='Real Price') #Close_actual_change
    plt.plot(df.Close_prediction, color='purple', label='Predicted Price') #Close_prediction_change
    plt.title(f'{ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    cwd = os.getcwd()
    path = cwd + f'\\plots_{date}'
    Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(cwd + f'\\plots_{date}\\plot_{ticker}_daily.png')
    plt.savefig(path + f'\\plot_{ticker}_{df_type}.png')
    # plt.show()
    plt.close()

    if change == 'absolute':
        plt.figure(figsize=(12, 6))
        plt.plot(pd.concat([df['Close_actual'], df['Added_changes']], axis=1))
        plt.title('Close Prediction (OHLC unchanged with SMA)')
        Path(path).mkdir(parents=True, exist_ok=True)
        # plt.savefig(cwd + f'\\plots_{date}\\absolute_change_{ticker}.png')
        plt.savefig(path + f'\\close_pred_{ticker}_{df_type}.png')
        plt.close()

    else:
        pass

def plot_loss(my_model, ticker, date = datetime.today().strftime('%d.%m')):
    plt.figure(figsize=(10, 6))
    print('history', my_model.history_)
    # print(my_model.history_.history)
    # plt.plot(my_model.history_['loss'], color='red') #history_
    plt.plot(my_model.history_['mean_squared_error'], color='green')
    plt.plot(my_model.history_['val_mse'], color='purple')
    # plt.plot(my_model.history_['mean_absolute_percentage_error'], color='purple')
    # plt.plot(my_model.history_['cosine_proximity'], color='blue')
    cwd = os.getcwd()
    path = cwd + f'\\loss_plot_with_SMA_{date}'
    Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(cwd + f'\\loss_plot_{date}\\plot_loss_{ticker}.png')
    plt.savefig(path + f'\\plot_loss_{ticker}.png')
    plt.close()

def plot_train_val(grid_result, ticker, date = datetime.today().strftime('%d.%m')):
    test_scores = -grid_result.cv_results_['mean_test_score']
    train_scores = -grid_result.cv_results_['mean_train_score']
    plt.plot(test_scores, label='test')
    plt.plot(train_scores, label='train')
    plt.legend(loc='best')
    cwd = os.getcwd()
    path = cwd + f'\\train_val_plot_with_SMA_{date}'
    Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(cwd + f'\\loss_plot_{date}\\plot_loss_{ticker}.png')
    plt.savefig(path + f'\\plot_loss_{ticker}.png')
    plt.close()
