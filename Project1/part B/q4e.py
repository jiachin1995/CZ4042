#
# Project 1,  part b
#

seed = 10
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed)

import pylab as plt

from tensorflow.keras.layers import Dense



#settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

test_filepath = 'datasets/testingdata.csv'
NUM_FEATURES = 7

learning_rate = 0.001
batch_size = 8
weight_decay = 0.001



#read testing data
admit_data = np.genfromtxt(test_filepath, delimiter= ',')
X_test, Y_test = admit_data[1:,1:8], admit_data[1:,-1]
Y_test = Y_test.reshape(Y_test.shape[0], 1)

#normalise inputs
X_test = (X_test- np.mean(X_test, axis=0))/ np.std(X_test, axis=0)

# load the model
model = tf.keras.models.load_model('models/3layers7features.h5')

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse']) 
              
loss, mse = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)

perf_list = []
perf_list.append(mse)



# load the model
model = tf.keras.models.load_model('models/4layers7features.h5')

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse']) 
       

#evaluate       
loss, mse = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)

perf_list.append(mse)



# load the model
model = tf.keras.models.load_model('models/4layers7features_wDropouts_1000epochs.h5')

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse']) 
       

#evaluate       
loss, mse = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)

perf_list.append(mse)

# load the model
model = tf.keras.models.load_model('models/5layers7features.h5')

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse']) 
       

#evaluate       
loss, mse = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)

perf_list.append(mse)

# load the model
model = tf.keras.models.load_model('models/5layers7features_wDropouts_1000epochs.h5')

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse']) 
       

#evaluate       
loss, mse = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)

perf_list.append(mse)
 
labels = [
    "3",
    "4",
    "4 with dropouts & 1000 epochs",
    "5",
    "5 with dropouts & 1000 epochs",
]
    
plt.bar(labels, perf_list)
plt.title('performance with reduced features')
plt.ylabel('mse loss')
plt.xlabel('layers')

plt.savefig('figures/q4e.png')
plt.show()