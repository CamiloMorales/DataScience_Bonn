
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np


# In[7]:

def init(config_networks):
    
    graph1 = tf.Graph()
    with graph1.as_default():
        
        x = tf.placeholder(tf.float32, shape=[None, config_networks["input-dim"]])
        y = tf.placeholder(tf.float32, shape=[None, config_networks["output-dim"]])
        
        '''
        keep_probs = {}
        for idx, config in enumerate(config_networks['configs']):
            keep_probs['network_{0}'.format(idx)] = tf.placeholder(tf.float32, shape=[len(config)])
        '''
        
        def create_variable(name, shape):
            return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))

        def create_mlp(config, x, input_dim):
            for idx,layer in enumerate(config):
                with tf.variable_scope("layer_{0}".format(idx)):
                    W = create_variable("weights", [input_dim, layer["num-units"]])
                    b = create_variable("biases", [layer["num-units"]])
                    
                    x = getattr(tf.nn, layer["trans-func"])(tf.matmul(x, W) + b, name='activation')
            
                    input_dim = layer["num-units"]
            return x
    
        losses = {}
        for idx, config in enumerate(config_networks["configs"]):
            with tf.variable_scope("network_{0}".format(idx)):
                h = create_mlp(config, x, config_networks['input-dim'])
                losses["network_{0}".format(idx)] = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))
                
        train_steps = [tf.train.AdamOptimizer(1e-4).minimize(loss) for loss in losses.values()]
        saver = tf.train.Saver()
        
        def train(data):
            with tf.Session(graph=graph1) as session:
                session.run(tf.initialize_all_variables())
                
                '''
                probs = {}
                for idx, network in enumerate(config_networks):
                    probs['network_{0}'.format(idx)] = [layer['keep-prob'] for layer in network]
                '''
                
                start = 0
                for i in xrange(config_networks['max-steps']):
                    batch_x = data["x"][start:(start+config_networks['train-batch-size']), :]
                    batch_y = data["y"][start:(start+config_networks['train-batch-size']), :]
                    start = (start + config_networks['train-batch-size']) % data["x"].shape[0]
                    session.run(train_steps, feed_dict={x:batch_x, y:batch_y})
                saver.save(session, config_networks['model-param-file'])
        
        def predict(data):
            results = {}
            with tf.Session(graph=graph1) as session:
                saver.restore(session, config_networks['model-param-file'])
                for idx,network in enumerate(config_networks["configs"]):
                    results["network_{0}".format(idx)] = []
                    for x_test in data:
                        x_test = np.reshape(x_test, [1,x_test.shape[0]])
                        results["network_{0}".format(idx)].append(graph1.get_tensor_by_name(
                                '{0}/{1}/activation:0'.format('network_{0}'.format(idx), 
                                        'layer_{0}'.format(len(network)-1))).eval(feed_dict={x:x_test}))
            return results
        
        return {
            'predict': predict,
            'train': train
        }


# In[ ]:



