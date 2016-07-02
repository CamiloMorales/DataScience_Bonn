
# coding: utf-8

# In[1]:

import numpy as np


# In[1]:

def convert_one_hot_encoding(y, number_of_classes):
    oneHot = np.zeros([y.shape[0], number_of_classes])
    for idx, l in enumerate(y):
        oneHot[idx, np.int32(l)] = 1
    return oneHot


# In[ ]:

def estimate_cross_entropy(labels, predicted_labels, epsilon = 0.0):
    one_hot = convert_one_hot_encoding(labels, np.unique(labels))
    return np.sum(np.multiply(one_hot,np.log(predicted_labels + epsilon))) * (-1.0/labels.shape[0]) 

