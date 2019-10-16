#
# Project 1 , part a
#

seed = 10
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed)

import pylab as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

#settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

train_filepath = 'datasets/trainingdata.csv'
test_filepath = 'datasets/testingdata.csv'

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 350
#optimal batch size
batch_size = 16
num_neurons = 10
weight_decay = 0


#read training data
train_input = np.genfromtxt(train_filepath, delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read testing data
test_input = np.genfromtxt(test_filepath, delimiter= ',')
testX, test_Y = test_input[1:, :21], test_input[1:,-1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix

# Create the model
# default kernel_initializer='glorot_uniform' , bias_initializer='zeros'
model = Sequential()
#hidden layer 1
model.add(
    Dense(
        units = num_neurons,
        input_shape = (NUM_FEATURES,),
        use_bias = True,
        activation = 'relu',
        kernel_regularizer = regularizers.l2(weight_decay)     #weight regularizers
    )
)
#output layer
model.add(
    Dense(
        units = NUM_CLASSES,
        use_bias = True,
        activation = 'softmax',
        kernel_regularizer = regularizers.l2(weight_decay)     #weight regularizers
    )
)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(trainX, trainY,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data = (testX, testY),
                    shuffle= True)

#print(model.layers[0].get_weights())
plt.figure(0)
plt.plot(range(epochs),history.history['acc'])
plt.plot(range(epochs),history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('figures/q4c.png')

