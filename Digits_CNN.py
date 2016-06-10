
# coding: utf-8

# In[ ]:

import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv


# In[ ]:

train = pd.read_csv('./train.csv').as_matrix().astype(np.float32)

np.random.shuffle(train)
#val = train[0:12000,:]
#train = train[12000:,:]


# In[ ]:

def convertOneHotEncoding(y):
    oneHot = np.zeros([y.shape[0], 10])
    for idx, l in enumerate(y):
        oneHot[idx, l] = 1
    return oneHot


# In[ ]:

x_train = train[:,1:].astype(float)
x_train = np.divide(x_train, 255.0)
train_mean = np.mean(x_train, axis=0)
train_mean = train_mean.reshape(1,784)
for idx,x in enumerate(x_train):
    x_train[idx] = x - train_mean

y_train = convertOneHotEncoding(train[:,0]).astype(np.float32)

#x_val = val[:,1:].astype(float)
#y_val = convertOneHotEncoding(val[:,0]).astype(np.float32)

#for idx,x in enumerate(x_val):
#    x_val[idx] = x - train_mean


# In[ ]:

test = pd.read_csv('./test.csv')
x_test = np.divide(test.as_matrix().astype(np.float32), 255.0)

for idx, x in enumerate(x_test):
    x_test[idx] = x - train_mean


# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


# In[ ]:

def createVariable(name, shape):
    return tf.get_variable(name, initializer = tf.truncated_normal(shape,stddev=0.1))
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:

x_image = tf.reshape(x, [-1,28,28,1])
keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope("layer1"):
    W_1 = createVariable("weights", [5,5,1,8])
    b_1 = createVariable("biases", [8])
    h_1 = tf.nn.relu(conv2d(x_image, W_1) + b_1)
    
with tf.variable_scope("layer2"):
    W_2 = createVariable("weights", [5,5,8,16])
    b_2 = createVariable("biases", [16])
    h_2 = tf.nn.relu(conv2d(h_1, W_2) + b_2)
    
with tf.variable_scope("layer3"):
    W_3 = createVariable("weights", [5,5,16,24])
    b_2 = createVariable("biases", [24])
    h_3 = tf.nn.relu(conv2d(h_2, W_3) + b_2)
    
h_4 = max_pool_2x2(h_3)

with tf.variable_scope("layer5"):
    W_5 = createVariable("weights", [5,5,24,24])
    b_5 = createVariable("biases", [24])
    h_5 = tf.nn.relu(conv2d(h_4,W_5) + b_5)
    
with tf.variable_scope("layer6"):
    W_6 = createVariable("weights", [5,5,24,24])
    b_6 = createVariable("biases", [24])
    h_6 = tf.nn.relu(conv2d(h_5,W_6) + b_6)
    
with tf.variable_scope("layer7"):
    W_7 = createVariable("weights", [5,5,24,24])
    b_7 = createVariable("biases", [24])
    h_7 = tf.nn.relu(conv2d(h_6,W_7) + b_7)
    
h_8 = max_pool_2x2(h_7)

with tf.variable_scope("layer9"):
    W_9 = createVariable("weights", [3,3,24,32])
    b_9 = createVariable("biases", [32])
    h_9 = tf.nn.relu(conv2d(h_8,W_9) + b_9)
    
with tf.variable_scope("layer10"):
    W_10 = createVariable("weights", [3,3,32,40])
    b_10 = createVariable("biases", [40])
    h_10 = tf.nn.relu(conv2d(h_9,W_10) + b_10)
    
with tf.variable_scope("layer11"):
    W_11 = createVariable("weights", [3,3,40,48])
    b_11 = createVariable("biases", [48])
    h_11 = tf.nn.relu(conv2d(h_10,W_11) + b_11)
    
h_12 = tf.reshape(max_pool_2x2(h_11), [-1, 4*4*48])

with tf.variable_scope("layer12"):
    W_12 = createVariable("weights", [4*4*48, 512])
    b_12 = createVariable("biases", [512])
    h_13 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_12, W_12) + b_12), keep_prob)
    
with tf.variable_scope("layer13"):
    W_o = createVariable("weights", [512, 10])
    b_o = createVariable("biases", [10])
    h = tf.nn.softmax(tf.matmul(h_13, W_o) + b_o)
    


# In[ ]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = tf.argmax(h,1)
saver = tf.train.Saver()


# In[ ]:

start = 0
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(25000):
        batch_x = x_train[start:(start+42),:]
        batch_y = y_train[start:(start+42),:]
        start = (start+42)%42000

        if i%500 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            save_path = saver.save(sess, "/home/camilo/kaggle_DataScience_Bonn/DataScience_Bonn.git/Digits/model/model.ckpt")
            print("Model saved in file: %s" % save_path)

        else:
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
    
    save_path = saver.save(sess, "/home/camilo/kaggle_DataScience_Bonn/DataScience_Bonn.git/Digits/model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    
    #print sess.run(accuracy, feed_dict={x:x_val, y:y_val})
    #np.savetxt('result.txt',sess.run(prediction, feed_dict={x:x_test, keep_prob:1.0}), fmt='%d')


# In[ ]:

temp_list = []

with tf.Session() as sess: 
      saver.restore(sess, "/home/camilo/kaggle_DataScience_Bonn/DataScience_Bonn.git/Digits/model/model.ckpt")
      for test in x_test:
            test = np.reshape(test, [1,784])
            temp_list.append(prediction.eval(feed_dict={x:test, keep_prob:1.0}))


# In[ ]:

#res = np.loadtxt('result.txt')

lis = []
for idx, y in enumerate(temp_list):
    lis.append([idx+1,y])
lis = np.array(lis)
np.savetxt('result2.txt', lis, fmt='%d', delimiter=',')


# In[ ]:



