
# coding: utf-8

# In[ ]:

import numpy as np
import copy
import json
import math
from sklearn import svm


# In[ ]:

def init(params, trainSet, trainLabel, valSet, valLabel):

    def Parse(in_params, out_params_dict, out_params_list):
        if 'level' not in in_params:
            out_params_list.append(out_params_dict)
            return 0
        level = in_params["level"]
        for value in level["values"]:
            out_params = copy.deepcopy(out_params_dict)
            out_params[level["param-name"]] = value["name"]
            Parse(value, out_params, out_params_list)
            
    def ParseParams(in_params):
        out_params_dict = {}
        out_params_list = []
        Parse(in_params, out_params_dict, out_params_list)
        return out_params_list
            
    def SVM(decision_function_shape = None):
        svm_params = params["SVM"]
        param_configs = ParseParams(svm_params)
        best_accuracy = float('-inf')
        best_config = {}
        for config in param_configs:
            config['decision_function_shape'] = decision_function_shape
            model = svm.SVC(**config)
            model.fit(trainSet, trainLabel)
            accuracy = model.score(valSet, valLabel)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        return best_config
        
    def MLP():
        return 1
    def RandomForest():
        return 1
    def DecisionTree():
        return 1
    def RBFNet():
        return 1
    return {
        "svm": SVM,
        "mlp": MLP,
        "random-forest": RandomForest,
        "decision-tree": DecisionTree,
        "rbf-net":RBFNet
    }


# In[ ]:




# In[ ]:



