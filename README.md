# Forecasting stock closing price using LSTM

Predicts close price for the list of the stocks taking stock prices for previous days (step size).

Different types of inputs were used in the model including
- Close price for the given time step (here for 30 days)
- Open, High, Low, Close prices for the given time step
- Absolute change of Close price for the given time step
- Absolute changes of Open, High, Low, Close prices for the given time step
- Percentage change of Close price for the given time step
- Percentage changes of Open, High, Low, Close prices for the given time step

Model performs GridSearch cross validation in order to find best parameters


For having final results run [main.py](https://github.com/gabsona/LSTM-final/blob/main/main.py) . 
Final output includes
- csv file with scores for each stock
- loss plot, close price prediction plots
- csv files with predictions for each day

## Experiments done


- Inputs - Open High Low Close prices are kept the same only Close price is changed into absolute change, output - Close absolute change [Results #1](https://github.com/gabsona/LSTM-final/blob/main/dict_only_close_change09.08.csv) [Results #2](https://github.com/gabsona/LSTM-final/blob/main/dict_only_close_change05.08.csv)
- Inputs - Open High Low Close prices are used to predict Close price, output - Close  [Results #1](https://github.com/gabsona/LSTM-final/blob/main/dict_close_18.08.csv), [Results #2](https://github.com/gabsona/LSTM-final/blob/main/dict_close_17.08.csv), [Results #3](https://github.com/gabsona/LSTM-final/blob/main/dict_close_16.08.csv), [Results #4](https://github.com/gabsona/LSTM-final/blob/main/dict_close_12.08.csv), [Results #5](https://github.com/gabsona/LSTM-final/blob/main/dict_close_11.08.csv)
- In training part _KerasClassifier_ was used instead of _KerasRegressor_ for solving classification problem (finding stock change directions). [Score results](https://github.com/gabsona/LSTM-final/blob/main/dict_clf_21.08.csv) - LSTM  classification problem without cross validation [Score results](https://github.com/gabsona/LSTM-final/blob/main/dict_clf_01.09.csv)


## All experiments

- 30 days are taken to predict 31st day for 3 years (training), for testing 30 days to predict 31st for 1 year
- only close price is used as an input
- open high low close are used as an input
- prediction done with price absolute or percentage changes as an input to predict absolute/percentage close change
- open high low prices are kept the same while instead of close price absolut change of close price is used
- input is hourly data instead of daily
- MAPE is used as a loss metric in addition to MSE
- technical indicators are added
- different activation functions are added in different layers
- different number of layers, number of units, dropout rates are being used
- different optimizers are used
- experiment with learning rates
- classification is done on the results of regression
- loss is calculated after each batch not epoch
