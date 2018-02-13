import tensorflow as tf
import sys

x = tf.Variable(0)
y = tf.Variable(1)

z = tf.multiply(x, y)

assign = tf.assign(x, 1)

print('Without control dependencies :')

#### Without control dependencies ####
with tf.Session() as sess:
    for _ in range(100):
        sess.run(tf.global_variables_initializer())
        sys.stdout.write(str(sess.run([assign, z])))


print('\n\nWith control dependencies :')

#### With control dependencies ####
with tf.control_dependencies([z]):
    assign = tf.assign(x, 1)

with tf.Session() as sess:
    for _ in range(100):
        sess.run(tf.global_variables_initializer())
        sys.stdout.write(str(sess.run([assign, z])))

print('')




