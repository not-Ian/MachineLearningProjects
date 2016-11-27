# -*- coding: utf-8 -*-

'''
Little example to let RNN's do addition

Made on June 12th 2016

Inspiration after Rajiv Shah's rnn_addition
http://projects.rajivshah.com/blog/2016/04/05/rnn_addition/
@author: robt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from tensorflow.python.ops       import clip_ops

from tensorflow.models.rnn       import seq2seq
from tensorflow.models.rnn       import rnn, rnn_cell

"""Hyperparameters"""
hidden_size    = 60     #hidden size of the LSTM
input_size     = 1      # For later expansion
batch_size     = 10    # batch_size
seq_len        = 10     # How long do you want the vectors with integers to add be?
drop_out       = 0.8    # Drop out
num_layers     = 2      # Number of RNN layers
max_iterations = 10000  # Number of iterations to train with
plot_every     = 10     # How often do you want terminal output?

def generate_data(upper, D, N):  # (5, seq_len, batch_size)
    """Function to data for rnn addition model.
    takes in:
    - upper: the upper margin of the integers to be added
    - D:     the dimensionality of the vectors (sequence length)
    - N:     the number of samples
    returns
    - X: the data
    - y: the labels"""

    X = np.zeros((N, D, 1))
    y = np.zeros(N)
    for i in range(N):
        X[i] = np.random.randint(0, upper, size=(D,1))
        print("{}: {}".format("np.random.randint(0, upper, size=(D,1))", X))
        y[i] = np.sum(X[i])
        print("{}: {}".format("np.sum(X[i])", y))
    return(X, y)

with tf.name_scope("Placeholders") as scope:
    inputs    = [tf.placeholder(tf.float32, shape=[batch_size, 2]) for _ in range(seq_len)]
    target    = tf.placeholder(tf.float32, shape=[batch_size])
    keep_prob = tf.placeholder("float")

with tf.name_scope("Cell") as scope:
    cell = rnn_cell.BasicLSTMCell(hidden_size)
    cell = rnn_cell.MultiRNNCell([cell] * num_layers)
    cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    initial_state = cell.zero_state(batch_size, tf.float32)

with tf.name_scope("RNN") as scope:
    outputs, states = seq2seq.rnn_decoder(inputs, initial_state, cell)
    final = outputs[-1]

with tf.name_scope("Output") as scope:
    W_o = tf.Variable(tf.random_normal([hidden_size, input_size], stddev=0.01))
    b_o = tf.Variable(tf.random_normal([input_size], stddev=0.01))
    prediction = tf.matmul(final, W_o) + b_o

with tf.name_scope("Optimization") as scope:
    cost = tf.pow(tf.sub(tf.reshape(prediction, [-1]), target),2)
    train_op = tf.train.RMSPropOptimizer(0.0005, 0.2).minimize(cost)
    loss = tf.reduce_sum(cost)

# Validation Data
X_val, y_val = generate_data(5, seq_len, batch_size)
X_val = np.split(np.squeeze(X_val), seq_len, axis=1)

# Fire up session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Collect information
perf_collect = np.zeros((int(np.floor(max_iterations /plot_every)),2))
step = 0
for k in range(1, max_iterations):

    #GenerateData for each iteration
    X, y = generate_data(5, seq_len, batch_size)
    X = np.split(np.squeeze(X), seq_len, axis=1)

    # Create the dictionary of inputs to feed into sess.run
    train_dict = {inputs[i]:X[i] for i in range(seq_len)}
    train_dict.update({target: y, keep_prob:drop_out})

    _, cost_train = sess.run([train_op, loss], feed_dict = train_dict) # perform an update on the paramters

    if (k%plot_every == 0): #Output information
        val_dict = {inputs[i]:X_val[i] for i in range(seq_len)} # create validation dictionary
        val_dict.update({target: y_val, keep_prob:1.0})
        cost_val = sess.run(loss, feed_dict = val_dict)         # compute the cost on the validation set
        perf_collect[step,0] = cost_train
        perf_collect[step,1] = cost_val

        print('At %.0f of %.0f train is %.2f val is %.2f'%(k, max_iterations, cost_train, cost_val))
        step += 1
result = sess.run([prediction], feed_dict=val_dict)

for i in range(batch_size):
    print('predicted %.2f true %.0f'%(result[0][i],y_val[i]))

plt.plot(perf_collect[:,0],label='train cost')
plt.plot(perf_collect[:,1],label='val cost')
plt.legend()
plt.show()
