
# coding: utf-8

# In[ ]:

import numpy as np

def init():

    def z_score_compute(data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return mean, std
    
    def z_score_normalize(data, mean, std):
        for idx,x in enumerate(data):
            data[idx] = np.divide(np.subtract(x, mean), std)


        

