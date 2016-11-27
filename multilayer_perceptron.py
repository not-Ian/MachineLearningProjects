'''

A Multilayer Perceptron implementation example using Tensorflow library.
This example is using the MNIST database of handwritten digits.
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Input MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Parameters
learning_rate   = 0.001
training_epochs = 100
batch_size      = 100
display_step    = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer number of features
n_hidden_4 = 256 # 4th layer number of features
n_input    = 784 # MNIST data input (img shape: 28*28)
n_classes  = 10  # MNIST total classes(0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create a model
def multilayer_perceptron(x, weights, biases):
    layer_1   = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1   = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2   = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2   = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3   = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3   = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4   = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4   = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1':  tf.Variable(tf.random_normal([n_input,    n_hidden_1])),
    'h2':  tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3':  tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4':  tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}

biases = {
    'b1':  tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':  tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':  tf.Variable(tf.random_normal([n_hidden_3])),
    'b4':  tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%01d' % (epoch+1), "cost=", \
                  "{:.9f}".format(avg_cost))
            if epoch =< 5:
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print ("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
            else if epoch % 5 == 0:
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print ("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
                
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
            
    print("Optimization Finished!")

    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
                     
