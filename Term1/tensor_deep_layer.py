# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 00:39:10 2018

@author: bakhtiar
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1

n_input = 784
n_classes = 10

n_hidden_layer = 256

weights = {
        'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}

biases = {
        'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

layer_l = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])

layer_l = tf.nn.relu(layer_l)

logits = tf.add(tf.matmul(layer_l, weights['out']), biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, label=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})