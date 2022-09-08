# Forecasting stock closing price using LSTM

Predicts close price for the list of the stocks taking stock prices for previous days (step size).

Different types of inputs were used in the model including
- Close price for the given time step (here for 30 days)
- Open, High, Low, Close prices for the given time step
- Absolute change of Close price for the given time step
- Absolute changes of Open, High, Low, Close prices for the given time step
- Percentage change of Close price for the given time step
- Percentage changes of Open, High, Low, Close prices for the given time step



For having final results run `main.py` . 
Final output includes
- csv file with scores for each stock
- loss plot, close price prediction plots
- csv files with predictions for each day

## Experiments done

