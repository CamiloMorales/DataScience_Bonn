
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

class Individual(object):
    def __init__(self):
        self._individual = []
        self._fitness = np.inf

    def __repr__(self):
        return "<"+str(self._individual)+" , "+str(self._fitness)+">"
        
    @property
    def individual(self):
        return self._individual
    
    @individual.setter
    def setIndividual(self, value):
        self._individual = value
        
    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def setFitness(self, value):
        self._fitness = value


# In[ ]:



