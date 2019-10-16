#
# Project 1 , part a
#

seed = 10
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed)

import pylab as plt

import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback

#settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# Create the model
# default kernel_initializer='glorot_uniform' , bias_initializer='zeros'
def create_model(num_neurons):
    model = Sequential()
    # hidden layer 1
    model.add(
        Dense(
            units=num_neurons,
            input_shape=(NUM_FEATURES,),
            use_bias=True,
            activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)  # weight regularizers
        )
    )
    # output layer
    model.add(
        Dense(
            units=NUM_CLASSES,
            use_bias=True,
            activation='softmax',
            kernel_regularizer=regularizers.l2(weight_decay)  # weight regularizers
        )
    )

    return model

class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time() - self.epoch_time_start)

train_filepath = 'datasets/trainingdata.csv'

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 350
#batch_size result from q2
batch_size = 16
weight_decay = 1e-6
k_folds = 5

num_neurons_choices = [5, 10, 15, 20, 25]

#read training data
train_input = np.genfromtxt(train_filepath, delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#define log
train_accuracylog = []
test_accuracylog = []
test_losslog = []
timelog = []

#perform k fold cross validation and training
interval = len(trainX) // k_folds

for fold in range(k_folds):

    #split data into training set and validation set
    start, end = fold * interval, (fold + 1) * interval

    X_test, Y_test = trainX[start:end], trainY[start:end]
    X_train = np.append(trainX[:start], trainX[end:], axis=0)
    Y_train = np.append(trainY[:start], trainY[end:], axis=0)


    timelog_per_epoch = []
    train_accuracylog_= []
    test_accuracylog_ = []
    test_losslog_ = []

    fold_count = fold

    for i in num_neurons_choices:

        num_neurons = i

        print("Folds : ", fold_count , " No neurons : " , num_neurons)

        # Create model
        model = create_model(num_neurons)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Create time call back to record epoch timing
        time_callback = TimingCallback()

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data = (X_test, Y_test),
            shuffle= True,
            callbacks = [time_callback])

        # mean over epochs
        timelog_per_epoch.append(np.mean(time_callback.logs))

        train_accuracylog_.append(history.history['acc'])
        test_accuracylog_.append(history.history['val_acc'])
        test_losslog_.append(history.history['val_loss'])

    timelog.append(timelog_per_epoch)
    train_accuracylog.append(train_accuracylog_)
    test_accuracylog.append(test_accuracylog_)
    test_losslog.append(test_losslog_)


# plot cross-validation accuracies against the number of epochs
# np.mean(train_accuracylog, axis=0) - mean over 5 folds
average_trainaccuracy = np.mean(train_accuracylog, axis=0)
plt.figure(0)
for i in range(len(num_neurons_choices)):
    plt.plot(range(epochs), average_trainaccuracy[i])
plt.title('average train accuracy against epochs')
plt.ylabel('train accuracy')
plt.xlabel('epochs')
plt.legend(num_neurons_choices, loc='upper right')
plt.savefig('figures/q3_cvtrainaccuracy')

# np.mean(test_accuracylog, axis=0) - mean over 5 folds
average_testaccuracy = np.mean(test_accuracylog, axis=0)
plt.figure(1)
for i in range(len(num_neurons_choices)):
    plt.plot(range(epochs), average_testaccuracy[i])
plt.title('average test accuracy against epochs')
plt.ylabel('test accuracy')
plt.xlabel('epochs')
plt.legend(num_neurons_choices, loc='upper right')
plt.savefig('figures/q3_cvtestaccuracy')

# mean over epochs
plt.figure(2)
average_train_accuracy_per_epoch = np.mean(average_trainaccuracy, axis=1)
print("Average_train_accuracy_per_epoch {}".format(average_train_accuracy_per_epoch))
average_test_accuracy_per_epoch = np.mean(average_testaccuracy, axis=1)
print("Average_test_accuracy_per_epoch {}".format(average_test_accuracy_per_epoch))
plt.plot(num_neurons_choices, average_train_accuracy_per_epoch)
plt.plot(num_neurons_choices, average_test_accuracy_per_epoch)
plt.xticks(num_neurons_choices)
plt.title('average train and test accuracies against number of neurons in hidden layer')
plt.ylabel('average accuracy per epoch')
plt.xlabel('no neurons')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('figures/q3_averageaccuracyperepoch')

