
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:

def init(config_random_forests):

    def train(data):
        labels = data[:,0]
        train = data[:,1:]
        model = RandomForestClassifier(**config_random_forests)
        model.fit(train, labels)
        return model

    def predict(test_data, model):
        return model.predict(test_data)
    
    def predict_probabilities(test_data, model):
        return model.predict_proba(test_data)

    return {
        'train': train,
        'predict': predict,
        'predict_probabilities': predict_probabilities
    }


# In[ ]:



