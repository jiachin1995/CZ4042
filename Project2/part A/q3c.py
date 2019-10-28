#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 750
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1


    """Moved rescaling here"""
    data = (data - 0)/255.

    return data, labels_




def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1
    HIDDEN_C1_NEURONS = 60  #40,60
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, HIDDEN_C1_NEURONS], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([HIDDEN_C1_NEURONS]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
	
    """add conv2 & 300 layer"""
    #Conv 2
    HIDDEN_C2_NEURONS = 70  #50,70
    W2 = tf.Variable(tf.truncated_normal([5, 5, HIDDEN_C1_NEURONS, HIDDEN_C2_NEURONS], stddev=1.0/np.sqrt(HIDDEN_C1_NEURONS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([HIDDEN_C2_NEURONS]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    #Flatten
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
    
    #Fully connected 300 layer
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    linear = tf.matmul(pool_2_flat, W3) + b3
    
    #Softmax
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(linear, W4) + b4

    return logits


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)


    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    
    logits = cnn(x)

    # Declare loss & accuracy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        train_loss=[]
        test_acc= []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            #apply mini batch
            for j in range(len(trainX) // batch_size):
                train_step.run(feed_dict={x: trainX[j*batch_size:(j+1)*batch_size], y_: trainY[j*batch_size:(j+1)*batch_size]})
            
            train_loss.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                
            print('epoch', e, 'entropy', train_loss[-1], 'test acc', test_acc[-1])
            
        print(train_loss)
        print(test_acc)
        
    
    print(test_acc[-1])
    
    plt.figure(1)
    plt.plot(range(epochs), test_acc)
    plt.plot(range(epochs), train_loss)
    plt.xlabel(str(epochs) + ' epochs')
    plt.ylabel('Test accuracy & Train loss')
    plt.legend(['test accuracy', 'train loss'], loc='upper left')

    
    plt.savefig('figures/q3c.png')
    plt.show()



if __name__ == '__main__':
  main()
