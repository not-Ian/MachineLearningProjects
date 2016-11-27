'''
A Recurrent Neural Network implementation example using TensorFlow Library.

Author: Ian Cunningham
'''

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Parameters
training_iters = 1000
n_epochs       = 1000
batch_size     = 128
display_step   = 100
learning_rate  = 0.001

n_observations = 100
n_input        = 2   # Input data (Num + Num)
n_steps        = 28  # timesteps
n_hidden_1     = 256 # 1st layer number of features
n_hidden_2     = 256 # 2nd layer number of features
n_classes      = 1   # Output

X  = tf.placeholder("float", [None, n_input])
X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y  = tf.placeholder(tf.float32)

# Random input data
x1 = 100 * np.random.random_sample([100,])
x2 = 100 * np.random.random_sample([100,])
   
add = tf.add(x1, x2)
mul = tf.mul(X1, X2)

weights = {
    'hidden1': tf.Variable(tf.random_normal([n_input,    n_hidden_1])),
    #'hidden2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':     tf.Variable(tf.random_normal([n_hidden_1,  n_classes]))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'hidden2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out':     tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X1, _weights, _biases):

    # Layer 1.1
    layer_1 = tf.add(tf.matmul(_X1, weights['hidden1']), biases['hidden1'])
    layer_1 = tf.nn.relu(layer_1)
    # Layer 1.2
    # layer_1_2 = tf.add(tf.matmul(_X2, weights['hidden2']), biases['hidden2'])
    # layer_1_2 = tf.nn.relu(layer_1_2)
    # Hidden layer with RELU activation
    layer_2   = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

    output    = tf.nn.relu(layer_2)

    return output

pred         = RNN(X1, weights, biases)
cost         = tf.reduce_sum(tf.pow(pred - Y, 2)) / (n_observations - 1)
optimizer    = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y,1))

init     = tf.initialize_all_variables()
# initData = tf.initialize_variables(x1.all(), x2.all())

with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(init)

    # Fit all training data
    prev_training_cost = 0.0

    for epoch_i in range(n_epochs) :
        for (_x1) in x1:
            for (_x2) in x2:
                print("Input 1:")
                print(_x1)
                print("Input 2:")
                print(_x2)
                print("Add function: ")
                print(sess.run(add, feed_dict={X1: x1, X2: x2}))
                y =   sess.run(add, feed_dict={X1: x1, X2: x2})
                print(y)
                sess.run(optimizer, feed_dict={X: x, Y: y})
            

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        print(training_cost)

        if epoch_i % 20 == 0:
            ax.plot(X1, X2, pred.eval(
                feed_dict={X1: x1, X2: x2}, session=sess),
                    'k', alpha=epoch_i / n_epochs)
            fig.show()
            plt.draw()

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
