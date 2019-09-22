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




#read training data
admit_data = np.genfromtxt(train_filepath, delimiter= ',')
a = admit_data[1:,1:9]
a_T = np.transpose(a) 

#correlation
corr_matrix = np.corrcoef(a_T)

# Save Numpy array to csv
np.savetxt('figures/q2.csv', corr_matrix, delimiter=',', fmt='%f')
    
labels = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR ",
    "CGPA",
    "Research"
]
plt.plot(labels, corr_matrix[-1,:-1])
plt.title('Correlation Graph')
plt.ylabel('chance of admission')
plt.xlabel('')

#plt.savefig('figures/q2.png')
plt.show()