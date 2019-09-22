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

train_filepath = 'datasets/trainingdata.csv'
test_filepath = 'datasets/testingdata.csv'
NUM_FEATURES = 7

learning_rate = 0.001
batch_size = 8
weight_decay = 0.001
# num_neuron = 30
epochs = 100


#read training data
admit_data = np.genfromtxt(train_filepath, delimiter= ',')
X_train, Y_train = admit_data[1:,1:8], admit_data[1:,-1]
Y_train = Y_train.reshape(Y_train.shape[0], 1)

#read testing data
admit_data = np.genfromtxt(test_filepath, delimiter= ',')
X_test, Y_test = admit_data[1:,1:8], admit_data[1:,-1]
Y_test = Y_test.reshape(Y_test.shape[0], 1)

#normalise inputs
X_train = (X_train- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)
X_test = (X_test- np.mean(X_test, axis=0))/ np.std(X_test, axis=0)

# Create the model
model = tf.keras.Sequential()
#hidden layer 1
model.add(
    Dense(
        units = 10,                 #number of neurons
        input_shape= (NUM_FEATURES,),
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
#output layer
model.add(
    Dense(
        units = 1,
        use_bias=True,
        activation='linear',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers

    ))

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Configure a model for mean-squared error regression.
model.compile(optimizer=optimizer,
              loss='mse',       # mean squared error
              metrics=['mse'])  

history = model.fit(X_train, Y_train, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data = (X_test, Y_test),
    shuffle=True
)

model.save("models/3layers7features.h5")

X_test50 = X_test[:50]
Y_test50 = Y_test[:50]

predictions = model.predict(X_test50, 
    batch_size=1)
    
plt.plot(range(1,51), predictions)
plt.plot(range(1,51), Y_test50)
plt.title('predictions vs targets')
plt.ylabel('chance of admission')
plt.xlabel('test sets')
plt.legend(['predictions', 'target'], loc='lower left')

plt.savefig('figures/q1c.png')
plt.show()