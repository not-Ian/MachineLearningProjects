# A logistic regression learning algorithm example using TensorFlow library

import input_data
import numpy as np
from numpy import genfromtxt
#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
import tensorflow as tf


data = genfromtxt('data.csv',delimiter=',')  # Training data
data1, data2, data3, data4, data5 = tf.split(1,5,data)
test_data = data3
data = tf.concat(data1,data2)
data6 = tf.concat(data4,data5)
data = tf.concat(data,data6)
#test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data

x_train=np.array([ i[1::] for i in data])
y_train,y_train_onehot = convertOneHot(data)

x_test=np.array([ i[1::] for i in test_data])
y_test,y_test_onehot = convertOneHot(test_data)


#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)
A=data.shape[1]-1 # Number of features, Note first is y
B=len(y_train_onehot[0])
tf_in = tf.placeholder("float", [None, A]) # Features
tf_weight = tf.Variable(tf.zeros([A,B]))
tf_bias = tf.Variable(tf.zeros([B]))
tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

# Training via backpropagation
tf_softmax_correct = tf.placeholder("float", [None,B])
tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

# Train using tf.train.GradientDescentOptimizer
tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

# Add accuracy checking nodes
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print("...")
# Run the training
for i in range(300):
    sess.run(tf_train_step, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})

# Print accuracy
    result = sess.run(tf_accuracy, feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})
    print ("Run {},{}".format(i,result))





###Parameters
##learning_rate = 0.01
##training_epochs = 25
##batch_size = 100
##display_step = 1
##
### Training Data
##x = tf.placeholder("float", [None, 784])
##y = tf.placeholder("float", [None, 10])
##
### Create model
##
### Set model weights
##W = tf.Variable(tf.zeros([784, 10]))
##b = tf.Variable(tf.zeros([10]))
##
### Construct a linear model
##activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
##
### Minimize error using cross entropy
### Cross Entropy
##cost = -tf.reduce_sum(y*tf.log(activation))
### Gradient Descent
##optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
##
### Initializing the variables
##init = tf.initialize_all_variables()
##
###Launch the graph
##with tf.Session() as sess:
##    sess.run(init)
##
##    for epoch in range(training_epochs):
##        avg_cost = 0.
##        total_batch = int(mnist.train.num_examples/batch_size)
##        # Loop over all batches
##        for i in range(total_batch):
##            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
##            # Fit training using batch data
##            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
##            # Compute average loss
##            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
##        #Display logs per epoch step
##        if epoch % display_step == 0:
##            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
##            
##    print ("Optimization Finished!")
##
##    # Test model
##    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
##    # Calculate accuracy
##    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
##    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
