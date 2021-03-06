import numpy as np
import tensorflow as tf

#import MNIST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(6000) #6000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(400) #200 for testing

#Reshape images to one dimension (1D)
Xtr = np.reshape(Xtr, newshape=(-1, 28*28))
Xte = np.reshape(Xte, newshape=(-1, 28*28))

#tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

#Nearest neighbor calculation using L1 distance
# calculate L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
#Predict: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

#Initializing the variables
init = tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)

    #loop over test data
    for i in range(len(Xte)):
        #Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:]})
        #Get nearest neighbor class label and copare it to its true label
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
              "True Class:", np.argmax(Yte[i]))
        #Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print ("Done!")
    print ("Accuracy: ", accuracy)
        
