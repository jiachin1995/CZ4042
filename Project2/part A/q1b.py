#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import cv2



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 10
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

    return data, labels_

def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
	
    """add conv2 & 300 layer"""
    #Conv 2
    
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(50*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')

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

    return [conv_1, pool_1, conv_2, pool_2]



def main():
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)


    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    c1, s1, c2, s2 = cnn(x)

    N = len(testX)
    idx = np.arange(N)

    #tensorflow saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        np.random.shuffle(idx)
        testX, testY = testX[idx], testY[idx]
    
        saver.restore(sess, "models/q1a.ckpt")

        #test raw 1
        X = testX[0,:]
        X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)

        X = (X - 0)/255.
        X = np.expand_dims(X, axis=0)
        c1_out = c1.eval(feed_dict={x: X})
        s1_out = s1.eval(feed_dict={x: X})
        c2_out = c2.eval(feed_dict={x: X})
        s2_out = s2.eval(feed_dict={x: X})

        #resize and convert to integer
        c1_out = (c1_out[0,:,:,0])*255
        c1_out = np.expand_dims(c1_out, axis=-1).astype(int)
        s1_out = (s1_out[0,:,:,0])*255
        s1_out = np.expand_dims(s1_out, axis=-1).astype(int)
        c2_out = (c2_out[0,:,:,5])*255
        c2_out = np.expand_dims(c2_out, axis=-1).astype(int)
        s2_out = (s2_out[0,:,:,5])*255
        s2_out = np.expand_dims(s2_out, axis=-1).astype(int)
        
        
        #save images
        cv2.imwrite('figures/test1_raw.png',X_show)
        cv2.imwrite('figures/test1_c1_out.png',c1_out)
        cv2.imwrite('figures/test1_s1_out.png',s1_out)
        cv2.imwrite('figures/test1_c2_out.png',c2_out)
        cv2.imwrite('figures/test1_s2_out.png',s2_out)
        
        
        #test raw 2
        X = testX[1,:]
        X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)

        X = (X - 0)/255.
        X = np.expand_dims(X, axis=0)
        c1_out = c1.eval(feed_dict={x: X})
        s1_out = s1.eval(feed_dict={x: X})
        c2_out = c2.eval(feed_dict={x: X})
        s2_out = s2.eval(feed_dict={x: X})

        #resize and convert to integer
        c1_out = (c1_out[0,:,:,0])*255
        c1_out = np.expand_dims(c1_out, axis=-1).astype(int)
        s1_out = (s1_out[0,:,:,0])*255
        s1_out = np.expand_dims(s1_out, axis=-1).astype(int)
        c2_out = (c2_out[0,:,:,5])*255
        c2_out = np.expand_dims(c2_out, axis=-1).astype(int)
        s2_out = (s2_out[0,:,:,5])*255
        s2_out = np.expand_dims(s2_out, axis=-1).astype(int)
        
        
        #save images
        cv2.imwrite('figures/test2_raw.png',X_show)
        cv2.imwrite('figures/test2_c1_out.png',c1_out)
        cv2.imwrite('figures/test2_s1_out.png',s1_out)
        cv2.imwrite('figures/test2_c2_out.png',c2_out)
        cv2.imwrite('figures/test2_s2_out.png',s2_out)        


if __name__ == '__main__':
  main()
