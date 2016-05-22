
# coding: utf-8

# In[ ]:

import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv


# In[ ]:

train = pd.read_csv('./train.csv').as_matrix()
np.random.shuffle(train)
val = train[0:12000,:]
train = train[12000:,:]


# In[ ]:

def convertOneHotEncoding(y):
    oneHot = np.zeros([y.shape[0], 10])
    for idx, l in enumerate(y):
        oneHot[idx, l] = 1
    return oneHot


# In[ ]:

x_train = train[:,1:].astype(float)

train_mean = np.mean(x_train, axis=0)
train_mean = train_mean.reshape(1,784)
for idx,x in enumerate(x_train):
    x_train[idx] = x - train_mean

y_train = convertOneHotEncoding(train[:,0]).astype(np.float32)

x_val = val[:,1:].astype(float)
y_val = convertOneHotEncoding(val[:,0]).astype(np.float32)

for idx,x in enumerate(x_val):
    x_val[idx] = x - train_mean


# In[ ]:

test = pd.read_csv('./test.csv')
x_test = test.as_matrix().astype(np.float32)


# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


# In[ ]:

W = tf.Variable(tf.truncated_normal([784,10],stddev=0.1), name="Weights")
b = tf.Variable(tf.zeros([10]), name="bias")


# In[ ]:

y_ = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_+1e-37), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)
prediction = tf.argmax(y_,1)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[ ]:

start = 0
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(60000):
        batch_x = x_train[start:(start+30),:]
        #print batch_x
        #break
        batch_y = y_train[start:(start+30),:]
        #print sess.run(cross_entropy, feed_dict={x:batch_x, y: batch_y})
        train_step.run(feed_dict={x:batch_x, y: batch_y})
        start = (start+30)%30000
        #break
    print sess.run(accuracy, feed_dict={x:x_val, y:y_val})
    np.savetxt('result.txt',sess.run(prediction, feed_dict={x:x_test}), fmt='%d')
    #np.savetxt('result_probs.txt',sess.run(y_,feed_dict={x:x_test}))
    


# In[ ]:

res = np.loadtxt('result.txt')
lis = []
for idx, y in enumerate(res):
    lis.append([idx+1,y])
lis = np.array(lis)
np.savetxt('result2.txt', lis, fmt='%d', delimiter=',')


# In[ ]:



