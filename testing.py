##import tensorflow as tf
##
### 1x2 matrix
##matrix1 = tf.constant([[3., 3.]])
### 2x1 matrix
##matrix2 = tf.constant([[2.],[2.]])
##
##product = tf.matmul(matrix1, matrix2)
##product_rev = tf.matmul(matrix2, matrix1)
##
##sess = tf.Session()
###sess = tf.InteractiveSession()
##
##result = sess.run(product)
##print(result)
##
##sess.close()

################################################################################
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'. Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2., -1.]
