
# coding: utf-8

# In[1]:

import numpy as np


# In[1]:

def convert_one_hot_encoding(y):
    oneHot = np.zeros([y.shape[0], 6])
    for idx, l in enumerate(y):
        oneHot[idx, l] = 1
    return oneHot


# In[ ]:



