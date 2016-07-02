
# coding: utf-8

# In[ ]:

import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import json

import sys
sys.path.insert(0, '../ToolBox')

import CrossValidation, MLP, Helper, ZScore


# In[ ]:

train = pd.read_csv('./train.csv')
keys = ["OutcomeType", "AnimalType", "SexuponOutcome", "AgeuponOutcome", "Breed", "Color"]

# In[ ]:

print "OutcomeType"

dict_temp = {}

dict_temp["OutcomeType"] = np.array([val for val in train.OutcomeType.unique() if str(val) != 'nan'and str(val).lower() != 'unknown'])
dict_temp["OutcomeType"] = np.sort(dict_temp["OutcomeType"]) #To make easier the construction of submission.

# In[ ]:

#print "OutcomeSubtype"

#dict_temp["OutcomeSubtype"] = np.array([val for val in train.OutcomeSubtype.unique() if str(val) != 'nan'and str(val).lower() != 'unknown'])
#dict_temp["OutcomeSubtype"]


# In[ ]:

print "Breed"
dict_temp["Breed"] = np.array([val for val in train.Breed.unique() if str(val) != 'nan'and str(val).lower() != 'unknown'])
np.set_printoptions(threshold='nan')

# In[ ]:

print "AnimalType"
dict_temp["AnimalType"] = np.array([val for val in train.AnimalType.unique() if str(val) != 'nan'and str(val).lower() != 'unknown'])
np.set_printoptions(threshold='nan')

# In[ ]:

print "SexuponOutcome"
dict_temp["SexuponOutcome"] = np.array([val for val in train.SexuponOutcome.unique() if str(val) != 'nan' and str(val).lower() != 'unknown'])

# In[ ]:

print "AgeuponOutcome"
dict_temp["AgeuponOutcome"] = train.AgeuponOutcome.unique()

# In[ ]:

print "Color"
dict_temp["Color"] = np.array([val for val in train.Color.unique() if str(val) != 'nan' and str(val).lower() != 'unknown'])

dict_colors = {}

i = 1

for value in dict_temp["Color"]:
    if value not in dict_colors and '/' not in value:
        dict_colors[value] = i
        i+=1
    elif value not in dict_colors and  '/' in value:
        temp_val = value.split('/')
        temp = '{1}/{0}'.format(temp_val[0], temp_val[1]) 
        if temp not in dict_colors:
            dict_colors[value] = i
            i += 1

#print dict_colors
            
def getColorCode(colorValue):
    if colorValue in dict_colors:
        return dict_colors[colorValue]
    splits = colorValue.split('/')
    if len(splits) > 1: #Must be 2
        val = '{1}/{0}'.format(splits[0], splits[1])
        if val in dict_colors:
            return dict_colors[val]
    return 0

# In[ ]:

import math

# In[ ]:

def convertAgeToNum(age):
    if str(age).isdigit():
        return age
    elif str(age) == 'nan':
        return 0
    else:
        age = age.strip()
        age = age.lower()
        splits = age.split()
        if "year" in splits[1]:
            return float(splits[0]) * 365
        elif "month" in splits[1]:
            return float(splits[0]) * 30
        elif "week" in splits[1]:
            return float(splits[0]) * 7
        elif "day" in splits[1]:
            return float(splits[0])


# In[ ]:

train_transformed = []
for idx, row in train.iterrows():
    lis = []
    for key in keys:
        if key == "AgeuponOutcome":
            age_in_days = convertAgeToNum(row[key])
            lis.append(age_in_days)
        elif key == 'Color':
            if str(row[key]) != "nan" and str(row[key]).lower() != "unknown":
                lis.append(getColorCode(row[key]))
            else:
                lis.append(0)
        elif str(row[key]) != "nan" and str(row[key]).lower() != "unknown":            
            temp = np.where(dict_temp[key] == row[key])[0][0]+1
            lis.append(temp)
        else:
            lis.append(0)
    train_transformed.append(np.array(lis))


# In[ ]:

train_transformed = np.array(train_transformed, np.int16)
print train_transformed


# In[ ]:

#Train the model: RandomForest.

labels = [x[0] for x in train_transformed]
train = [x[1:] for x in train_transformed]


# In[ ]:

test = pd.read_csv('./test.csv')
test_keys = ["AnimalType", "SexuponOutcome", "AgeuponOutcome", "Breed", "Color"]

# In[ ]:

test_transformed = []
for idx, row in test.iterrows():
    lis = []
    for key in test_keys:
        if key == "AgeuponOutcome":
            age_in_days = convertAgeToNum(row[key])
            lis.append(age_in_days)
        elif key == 'Color':
            if str(row[key]) != "nan" and str(row[key]).lower() != "unknown":
                lis.append(getColorCode(row[key]))
            else:
                lis.append(0)
        elif str(row[key]) != "nan" and str(row[key]).lower() != "unknown":
            temp = np.where(dict_temp[key] == row[key])
            if len(temp[0]) > 0:
                lis.append(temp[0][0]+1)
            else:
                lis.append(0)
        else:
            lis.append(0)
    test_transformed.append(np.array(lis))


# In[ ]:

test_transformed = np.array(test_transformed, np.int16)
#np.set_printoptions(threshold='nan')
print test_transformed


# In[ ]:

#train = np.array(train)
#trainSet = train[:int(train.shape[0]*0.7), :]
#valSet = train[int(train.shape[0]*0.7):,:]

#labels = np.array(labels)
#trainLabelSet = labels[:int(labels.shape[0]*0.7)]
#valLabelSet = labels[int(labels.shape[0]*0.7):]


#train = np.array(train)
#trainSet = train[:70, :]
#valSet = train[71:100,:]

#labels = np.array(labels)
#trainLabelSet = labels[:70]
#valLabelSet = labels[71:100]


# In[ ]:

#import yaml

#with open("./ToolBox/params.json") as data_file:
    ##params = json.loads(data_file, object_hook=ascii_encode_dict)
#    params = yaml.safe_load(data_file)
    
#functions = CrossValidation.init(params, trainSet, trainLabelSet, valSet, valLabelSet)

#best_config = functions["svm"](decision_function_shape="ovo")

#best_config


# In[ ]:

#from sklearn import svm

#rf = svm.SVC(decision_function_shape='ovo')
#rf.fit(train, labels)


# In[ ]:

#np.savetxt('./submission_SVM_OVO.csv', rf.predict(test_transformed), delimiter=',', fmt='%f')


# In[ ]:

#res = np.loadtxt('submission_SVM_OVO.csv', delimiter=',')

#lis = []
#for idx, y in enumerate(res):
#    temp_lis = np.zeros(6)
#    temp_lis[0] = idx+1
#    temp_lis[y] = 1
#    lis.append(temp_lis)

#np.savetxt('submission_SVM_OVO.csv', lis, fmt='%d', delimiter=',')


# In[ ]:

import yaml

with open("../ToolBox/params.json") as data_file:
    params = yaml.safe_load(data_file)

dict_zscore = ZScore.init()

print "Starting normalization of train data."
train_mean, train_stddev = dict_zscore["z_score_compute"](train)
print "Mean and stddev calculated."
normalized_train_data = dict_zscore["z_score_normalize"](train, train_mean, train_stddev)
print "Normalized."

validation_mean, validation_stddev = dict_zscore["z_score_compute"](normalized_train_data)
print "validation_mean: "+str(validation_mean)
print "validation_stddev: "+str(validation_stddev)

print np.array(normalized_train_data).shape

data_dict={"y": Helper.convert_one_hot_encoding(np.array(labels)), "x":np.array(normalized_train_data)}

print data_dict["y"].shape
print params['MLP']["output-dim"]

print "Starting training."
dict_mlp = MLP.init(params['MLP'])
print "Training."
dict_mlp['train'](data_dict)
print "Training finished."

print "Starting normalization of test data."
normalized_test_data = dict_zscore["z_score_normalize"](test_transformed, train_mean, train_stddev)
print "Predicting with test data."
print normalized_test_data.shape
result_dict = dict_mlp['predict'](normalized_test_data)
print "Prediction:"
result_dict


# In[ ]:

lis = []

# for idx, z in enumerate(result_dict["network_0"]):
#     temp_str = str(idx+1)
#     for i in xrange():
#
#     for idy, x in enumerate(z.tolist()):
#         temp_str = temp_str+"- "+str(x)
#     lis.append(temp_str)
#     print np.array(lis)
#     break
    
#lis = np.array(lis)
#np.savetxt('result2.txt', lis, fmt='%d', delimiter=',')


# In[ ]:



