
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

def init(config_mlp):
    
    graph1 = tf.Graph()
    with graph1.as_default():
        
        x = tf.placeholder(tf.float32, shape=[None, config_mlp["input-dim"]])
        y = tf.placeholder(tf.float32, shape=[None, config_mlp["output-dim"]])
        
        '''
        keep_probs = {}
        for idx, config in enumerate(config_networks['configs']):
            keep_probs['network_{0}'.format(idx)] = tf.placeholder(tf.float32, shape=[len(config)])
        '''
        
        def create_variable(name, shape):
            return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))

        def create_mlp(config, x, input_dim, output_dim):
            for idx,layer in enumerate(config):
                with tf.variable_scope("layer_{0}".format(idx)):
                    W = create_variable("weights", [input_dim, layer["num-units"]])
                    b = create_variable("biases", [layer["num-units"]])
                    x = getattr(tf.nn, layer["trans-func"])(tf.matmul(x, W) + b, name='activation')
                    input_dim = layer["num-units"]
                    
            with tf.variable_scope('layer_{0}'.format('output')):
                W = create_variable('weights', [input_dim, output_dim])
                b = create_variable('biases', [output_dim])
                x = tf.nn.softmax(tf.matmul(x,W) + b, name='activation')
            return x
        
        h = create_mlp(config_mlp['layer-defs'], x, config_mlp['input-dim'], config_mlp['output-dim'])
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        saver = tf.train.Saver()
        
        def train(data):
            with tf.Session(graph=graph1) as session:
                session.run(tf.initialize_all_variables())
                
                start = 0
                for i in xrange(config_mlp['max-steps']):
                    batch_x = data["x"][start:(start+config_mlp['train-batch-size']), :]
                    batch_y = data["y"][start:(start+config_mlp['train-batch-size']), :]
                    start = (start + config_mlp['train-batch-size']) % data["x"].shape[0]
                    session.run(train_step, feed_dict={x:batch_x, y:batch_y})
                saver.save(session, config_mlp['model-param-file'])
        
        def predict(data, layer_num = None):
            results = []
                
            with tf.Session(graph=graph1) as session:
                saver.restore(session, config_mlp['model-param-file'])
                if layer_num is None:
                    layer_num = 'output'
                
                for x_test in data:
                    x_test = np.reshape(x_test, [1,x_test.shape[0]])
                    results.append(graph1.get_tensor_by_name('{0}/activation:0'.format('layer_{0}'.format(layer_num))).eval(feed_dict={x:x_test}))
            
            results = np.array(results)
            return np.reshape(results, [results.shape[0], results.shape[2]])
        
        return {
            'predict': predict,
            'train': train
        }


# In[ ]:



