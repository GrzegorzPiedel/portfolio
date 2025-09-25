# Time Series Forecasting: SP500 Prices with LSTM and GRU

This project demonstrates how to prepare and model time series data for stock price prediction using recurrent neural networks (RNNs), specifically LSTM and GRU.

## Overview

The project covers:

1. **Data Preprocessing**  
   - Loading SP500 historical prices.
   - Sorting data by date.
   - Scaling data using MinMaxScaler.
   - Creating sequential windows for model input.

2. **Modeling**  
   - Building LSTM and GRU models using Keras.
   - Adding Batch Normalization and Dropout to improve training stability and prevent overfitting.
   - Using KerasTuner to optimize hyperparameters like the number of neurons and dropout rate.

3. **Evaluation**  
   - Predicting future stock prices on test data.
   - Calculating RMSE (Root Mean Squared Error) for performance comparison.
   - Visualizing predictions against actual SP500 prices.

## Dependencies

- Python 3.8+
- Pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- KerasTuner
- Matplotlib
