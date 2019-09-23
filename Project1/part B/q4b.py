#
# Project 1,  part b
#

seed = 10
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed)

import pylab as plt

from tensorflow.keras.layers import Dense, Dropout



#settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

train_filepath = 'datasets/trainingdata.csv'
test_filepath = 'datasets/testingdata.csv'
NUM_FEATURES = 7

learning_rate = 0.001
batch_size = 8
weight_decay = 0.001
num_neuron = 50
epochs = 100

dropout_keep = 0.8


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


# Create the 4-layer model
model = tf.keras.Sequential()
#hidden layer 1
model.add(
    Dense(
        units = num_neuron,                 #number of neurons
        input_shape= (NUM_FEATURES,),
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
model.add(Dropout(rate= 1-dropout_keep , seed=seed))
#hidden layer 2
model.add(
    Dense(
        units = num_neuron,                 #number of neurons
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
model.add(Dropout(rate= 1-dropout_keep , seed=seed))
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

#model.save("models/4layers7features_wDropouts.h5")


    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig('figures/q4b1.png')

# Create the 5-layer model
model = tf.keras.Sequential()
#hidden layer 1
model.add(
    Dense(
        units = num_neuron,                 #number of neurons
        input_shape= (NUM_FEATURES,),
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
model.add(Dropout(rate= 1-dropout_keep , seed=seed))
#hidden layer 2
model.add(
    Dense(
        units = num_neuron,                 #number of neurons
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
model.add(Dropout(rate= 1-dropout_keep , seed=seed))
#hidden layer 3
model.add(
    Dense(
        units = num_neuron,                 #number of neurons
        use_bias=True,
        activation='relu',
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)    #weight regularizers
    ))
model.add(Dropout(rate= 1-dropout_keep , seed=seed))
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

#model.save("models/5layers7features_wDropouts.h5")

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig('figures/q4b2.png')
plt.show()