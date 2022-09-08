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


For having final results run [main.py][https://github.com/gabsona/LSTM-final/blob/main/main.py] . 
Final output includes
- csv file with scores for each stock
- loss plot, close price prediction plots
- csv files with predictions for each day

## Experiments done

1. 
1. Inputs - Open High Low prices are kept the same only Close price is changed into absolute change, output - Close absolute change [Results #1](https://github.com/gabsona/LSTM-final/blob/main/dict_close_18.08.csv), [Results #2](https://github.com/gabsona/LSTM-final/blob/main/dict_close_17.08.csv), [Results #3](https://github.com/gabsona/LSTM-final/blob/main/dict_close_16.08.csv), [Results #4](https://github.com/gabsona/LSTM-final/blob/main/dict_close_12.08.csv), [Results #5](https://github.com/gabsona/LSTM-final/blob/main/dict_close_11.08.csv)
2. In training part _KerasClassifier_ was used instead of _KerasRegressor_ for solving classification problem (finding stock change directions). [Score results](https://github.com/gabsona/LSTM-final/blob/main/dict_clf_21.08.csv) 
2. LSTM  classification problem without cross validation [Score results](https://github.com/gabsona/LSTM-final/blob/main/dict_clf_01.09.csv)
3. 
