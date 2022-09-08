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

    if change == 'absolute':
        plt.figure(figsize=(12, 6))
        plt.plot(pd.concat([df['Close_actual'], df['Added_changes']], axis=1))
        plt.title('Close Absolute Change Prediction (only adding changes)')
        Path(path).mkdir(parents=True, exist_ok=True)
        # plt.savefig(cwd + f'\\plots_{date}\\absolute_change_{ticker}.png')
        plt.savefig(path + f'\\absolute_change_{ticker}_{df_type}.png')
        plt.close()

    else:
        pass

def plot_loss(my_model, ticker, date = datetime.today().strftime('%d.%m')):
    plt.figure(figsize=(10, 6))
    print(my_model.history)
    print(my_model.history.history)
    plt.plot(my_model.history.history['val_loss'], color='red') #history_
    # plt.plot(my_model.history_['mean_absolute_error'], color='green')
    # plt.plot(my_model.history_['mean_absolute_percentage_error'], color='purple')
    # plt.plot(my_model.history_['cosine_proximity'], color='blue')
    cwd = os.getcwd()
    path = cwd + f'\\loss_plot_{date}'
    Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(cwd + f'\\loss_plot_{date}\\plot_loss_{ticker}.png')
    plt.savefig(path + f'\\plot_loss_{ticker}.png')
