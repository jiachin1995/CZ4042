import numpy as np
import pandas as pd 

training_ratio = 0.7
seed = 10
np.random.seed(seed)

#read data 
admit_data = pd.read_csv('ctg_data_cleaned.csv', index_col=False)

labels = admit_data.columns.values
print("labels = {}".format(labels))
print("(row,cols) = {}".format(admit_data.shape))

data_size = admit_data.shape[0]
print("num of rows = {}".format(data_size))


#divide data into test and train sets 
idx = np.arange(data_size)  #create an index array
np.random.shuffle(idx)      #shuffle idx

#slice idx according to training_ratio
training_cutoff = int(data_size * training_ratio)
print("num of training sets = {}".format(training_cutoff))

train_idx, test_idx = idx[:training_cutoff], idx[training_cutoff:]
train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

#create training set
train = admit_data.iloc[train_idx]
train.to_csv("trainingdata.csv", index=False)

#create testing set
test = admit_data.iloc[test_idx]
test.to_csv("testingdata.csv", index=False)

