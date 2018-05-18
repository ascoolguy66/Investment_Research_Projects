# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:40:45 2017

@author: Veer Abhimanyu Singh
"""

# Stock Price Prediction using Neural Network

# Import libraries
import pandas as pd
import numpy as np
import math
import os
import pandas_datareader as pdr
import datetime
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import time
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model


# Set Working Directory
os.chdir("C:\\Users\\veer\\Desktop\\Projects\\Stock_Prediction")
output_path = "C:\\Users\\veer\\Desktop\\Projects\\Stock_Prediction"

# Creating a number of technical indicators
data = pd.read_csv("C:\\Users\\veer\\Desktop\\Projects\\Stock_Prediction\\IBM.csv")
dataset = data.iloc[:, [1,4]].values
holdout = dataset[1250:,:]
dataset = dataset[0:1250,:]

# Feature Scaling
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Sizes of dataset, train_ds, test_ds
dataset_size = X.shape[0]
train_size = X_train.shape[0]
test_size = X_test.shape[0]
 
# reshape our data into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (train_size, 1, 1))
y_train = np.reshape(y_train, (train_size, 1))

############ Building the RNN ############
# Initializing the RNN
regressor = Sequential()

# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=1000, return_sequences=True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(.2))

# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=500, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 3rd layer with some drop out regularization
regressor.add(LSTM(units=250, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 4th layer with some drop out regularization
regressor.add(LSTM(units=125, return_sequences=False))
regressor.add(Dropout(.2))
 
# Output layer
regressor.add(Dense(units=1, activation='sigmoid'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train
history = regressor.fit(X_train, y_train, epochs=20, batch_size=25)

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
inputs = np.reshape(inputs, (test_size, 1, 1))
predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price

# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# MSE
mse = mean_squared_error(predicted_stock_price[:, 0], predicted_stock_price[:, 1])
mse

################################ Testing on Hold-out dataset ###########################################
holdout_scaled = scaler.fit_transform(holdout)
inputs = np.array(holdout_scaled[:,0])
inputs = np.reshape(inputs, (105, 1, 1))

holdout_real_price = np.array(holdout_scaled[:,1])
holdout_predicted_price = regressor.predict(inputs)

# rebuild the Structure
dataset_pred_holdout = pd.DataFrame()
dataset_pred_holdout['real'] = holdout_real_price
dataset_pred_holdout['predicted'] = holdout_predicted_price

# real test data price VS. predicted price
holdout_prices = scaler.inverse_transform(dataset_pred_holdout)

# MSE
mse_holdout = mean_squared_error(holdout_prices[:, 0], holdout_prices[:, 1])
mse_holdout
############ Visualizing the results ############
## Visualising the results
plt.plot(holdout_prices[:, 0])
plt.plot(holdout_prices[:, 1])
plt.scatter(holdout_prices[:, 0], holdout_prices[:, 1])
plt.plot( [140,170],[140,170] )