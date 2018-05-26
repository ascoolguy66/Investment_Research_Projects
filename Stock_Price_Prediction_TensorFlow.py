
# coding: utf-8

# In[1]:


import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm


# In[2]:


# Get Data from Quandl: Microsoft
df = quandl.get("WIKI/MSFT")


# In[3]:


# Take A look at the data
# Look the oldest data
df.head()


# In[4]:


# Look at the latest data
df.tail()


# In[5]:


# Get only the adjusted close columns
df = df[['Adj. Close']]


# In[6]:


# Keeping 30 days recent data seperate from training and testing data to test the model
forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up


# In[7]:


# Converting Stock Price to numpy arrays, also removing data for the last 30 days
X = np.array(df.drop(['Prediction'], 1))
X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
X = preprocessing.scale(X)


# In[8]:


# Converting Predicted Price to numpy arrays, also removing data for the last 30 days
y = np.array(df['Prediction'])
y = y[:-forecast_out]
y = preprocessing.scale(y)


# In[9]:


# Splitting data into training and test datasets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


# In[10]:


# Import TensorFlow
import tensorflow as tf

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None])


# In[11]:


# Model architecture parameters
n_neurons_1 = 32
n_neurons_2 = 16
n_neurons_3 = 8
n_neurons_4 = 4
n_target = 1

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([1, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)


# In[12]:


# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Number of epochs and batch size
epochs = 10
batch_size = 4

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)


# In[14]:


# Predict for X_forecast
X_forecast_predicted = net.run(out, feed_dict={X: X_forecast})


# In[30]:


X_forecast = X_forecast.flatten()


# In[31]:


X_forecast_predicted = X_forecast_predicted.flatten()


# In[33]:


plt.plot(X_forecast, color="red") # Actual Data
plt.plot(X_forecast_predicted, color = "blue") # Forecasted Data
plt.show()

