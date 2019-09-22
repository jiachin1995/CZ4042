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


for i in range(NUM_FEATURES):
    modifier_list = [7/6,7/6,7/6,7/6,7/6,7/6,7/6]
    modifier_list[i] = 0
    
    #to remove a feature, we set the input to 0 & increase remaining features to account for lost weights 
    reduced_X_test = X_test * np.array(modifier_list)
    
    loss, mse = model.evaluate(x=reduced_X_test, y=Y_test, batch_size=batch_size)
    perf_list.append(mse)

print(perf_list)

    
labels = [
    "all features",
    "-GRE Score",
    "-TOEFL Score",
    "-University Rating",
    "-SOP",
    "-LOR",
    "-CGPA",
    "-Research",
]
    
plt.bar(labels, perf_list)
plt.title('performance with reduced features')
plt.ylabel('mse loss')
plt.xlabel('features')

#plt.savefig('figures/q3.png')
plt.show()