# Stock Price Prediction:
## Using Long Short Term Memory- RNN model.
## A sample program written in Python to implement LSTM- RNN based model for Stock Price Prediction.
* Input Data: IBM.csv, downloaded from Yahoo.com
* Python Code: Stock_Price_Prediction_Sample.py
* Actual.png: Shows plot of Actual price trend for hold out dataset
* Actual.png: Shows plot of Actual price trend for hold out dataset
* Predicted.png: Shows plot of Predicted price trend for hold out dataset
* Actual_Versus_Predicted.png: Actual versus predicted for hold out data set.

## Description: 
* Data for 05 years were used. The latest 05 months data was used as a holdout data set. The remaining data was used for training and testing the RNN model.
* The opening price as taken as predictior variable, and the Closing price was taken as target variable.
* The objective was to predict the closing price based on the opening price for a selected stock.
* A four layers LSTM-RNN model was built as follows:
``` 
 * First LSTM Layer: 1000 inputs and 1000 outputs.
 * Second LSTM Layer: 1000 inputs and 500 outputs
 * Third LSTM Layer: 500 inputs and 250 outputs
 * Fourth LSTM Layer: 250 inputs and 125 outputs
 * Output Layer: 125 inputs and 1 output with Sigmoid Activation Function.
 * Dropout rate is set at 20% at each LSTM layer.
``` 
 ## Using Artificial Neural Network on TensorFlow.
## A sample program written in Python to implement ANN-TensorFlow based model for Stock Price Prediction.
* Input Data: Downloaded from quandl for Microsoft
* Python Code: Stock_Price_Prediction_TensorFlow.py
* Jupyter Notebook: Stock_Price_Prediction_TensorFlow.ipynb. Includes all the outputs.

## Description: 
* Data used for 30 years. The goal was to predict stock price for 30 days.
* The price is taken as predictior variable, and the 30 days future price was taken as target variable.
* The objective was to predict the price for 30 days based on the current price.
* A four layers LSTM-RNN model was built as follows:
``` 
 * Input and First Hidden Layer: 1 input and 32 outputs.
 * Second Hidden Layer: 32 inputs and 8 outputs
 * Third Hidden Layer: 16 inputs and 4 outputs
 * Fourth Hidden Layer: 8 inputs and 4 outputs
 * Output Layer: 4 inputs and 1 output.
 * ReLU is used as an activation function for all the hidden layers.
```
