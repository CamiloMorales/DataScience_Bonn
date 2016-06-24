
# coding: utf-8

# In[ ]:

import numpy as np
import copy
import json
import math
from sklearn import svm
import time


# In[3]:

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
        
        global_start_time = time.time()
        
        svm_params = params["SVM"]
        param_configs = ParseParams(svm_params)
        
        print "Total configurations: "+ str(len(param_configs))
        count = 0
        
        best_accuracy = float('-inf')
        best_config = {}
        for config in param_configs:
            current_start_time = time.time()
            count+=1
            print "Starting configuration: "+ str(count) +", params: "+ str(config)
            config['decision_function_shape'] = decision_function_shape
            model = svm.SVC(**config)
            model.fit(trainSet, trainLabel)
            print "Model fit finished. Calculating accuracy."
            accuracy = model.score(valSet, valLabel)
            print "Current setting accuracy: "+ str(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
            print "Current best accuracy: "+ str(best_accuracy)
            current_elapsed_time_seconds = time.time() - current_start_time
            print("------ Current setting execution time: %d seconds = %d minutes ------" % (current_elapsed_time_seconds, current_elapsed_time_seconds/60) )
        total_elapsed_time_seconds = time.time() - global_start_time
        print("--- Total execution time: %s seconds = %s minutes ---" % (total_elapsed_time_seconds, total_elapsed_time_seconds/60) )
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

