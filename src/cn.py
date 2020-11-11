import tensorflow as tf
import numpy as np

x= np.random.randn(4,5,20)
input = tf.constant(x)
x = tf.cast(input,'float32')
con = tf.get_variable("weight",[20, 10])


z=tf.dot(x,con)
# z=tf.nn.conv2d(tf.cast(input,'float32'),con,strides=[1,1,1,1],padding="VALID")

sess=tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(z)

print(output.shape)