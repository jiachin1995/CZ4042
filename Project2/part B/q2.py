import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

batch_size = 128
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x):
  """ADD EMBEDDING LAYER"""
  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=no_words, embed_dim=20)

  input_layer = tf.expand_dims(word_vectors, -1)


  # print(input_layer.shape)     # (?,100,20,1)

  with tf.variable_scope('CNN_Layer1'):
    #layer 1
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=10,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')

    # print(conv1.shape)           # (?,81,1,10)
    # print(pool1.shape)           # (?,41,1,10)

    #layer2
    conv2 = tf.layers.conv2d(
        pool1,
        filters=10,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
   
    # print(conv2.shape)              # (?,22,1,10)   
    # print(pool2.shape)              # (?,11,1,10)   

    pool2_flat = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])       #reduce_max finds max number along an axis. Squeeze removes all dimension of size 1

    # print(pool2_flat.shape)         # (?,10) 

  """SOFTMAX LAYER, 15 CLASSES OUTPUT--- NO NEED DO"""
  logits = tf.layers.dense(pool2_flat, MAX_LABEL, activation=None)

  # print(logits.shape)               # (?,15) 
  

  return logits


def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      """ERROR FOUND! should be .append(row[2]) instead of row[1]"""
      x_train.append(row[2])        
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  #changed char to vocab processor.
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(vocab_processor.fit_transform(x_train)))
  x_test = np.array(list(vocab_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  
  #print(x_train.shape)  5600,100
  #print(y_train.shape)  5600
  
  # print(x_train)            
  # print(np.amax(x_train))       #38657
  global no_words
  no_words = len(vocab_processor.vocabulary_)
  # print(no_words)               #38658
  
  # print(y_test)         
  # print(y_test.shape) # (700,)
  
  """Note: No need to normalise x inputs. its one hot encoding"""

  
  return x_train, y_train, x_test, y_test

  
def main():
  
  x_train, y_train, x_test, y_test = read_data_chars()

  # print(len(x_train))   #5600
  # print(len(x_test))    #700

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits = char_cnn_model(x)

  # Optimizer
  labels = tf.one_hot(y_, MAX_LABEL)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  #accuracy
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)


  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  #for shuffling
  N = len(x_train)
  idx = np.arange(N)

  """Record running times"""
  import time
  
  start_time = time.time()
  # training
  train_loss = []
  test_acc= []
  for e in range(no_epochs):
    #shuffle
    np.random.shuffle(idx)
    x_train, y_train = x_train[idx], y_train[idx]

    #apply mini batch of 128
    for j in range(len(x_train) // batch_size):
        _, loss_  = sess.run([train_op, entropy], {
                    x: x_train[j*batch_size:(j+1)*batch_size], 
                    y_: y_train[j*batch_size:(j+1)*batch_size]
                })
        
    train_loss.append(loss_)
    acc_ = sess.run([accuracy], {
                    x: x_test, 
                    y_: y_test
                })
    test_acc.append(acc_)
        
    print('epoch', e, 'entropy', train_loss[-1], 'test acc', test_acc[-1])

  #get runtime
  run_time = time.time() - start_time      
  print(run_time) #in seconds
  
  """Plot <training entropy cost> & <test accuracy> over epochs"""
   
  plt.figure(1)
  plt.plot(range(no_epochs), test_acc)
  plt.plot(range(no_epochs), train_loss)
  plt.xlabel(str(no_epochs) + ' epochs')
  plt.ylabel('Test accuracy & Train loss')
  plt.legend(['test accuracy', 'train loss'], loc='upper left')

    
  plt.savefig('figures/q2.png')
  plt.show()
  
  sess.close()

if __name__ == '__main__':
  main()
