import tensorflow as tf

#### If-else-clause ####
x = tf.constant(1.)

condition = x > 0

result = tf.cond(condition, lambda: tf.sqrt(x), lambda: 0.)

with tf.Session() as sess:
    print(sess.run(result))


tf.reset_default_graph()

#### While loop ####
num_iterations = tf.constant(10, name='Num_of_iterations')
loop_step = tf.constant(1, name='Loop_step')
value_init = tf.constant(1., name='Value_init')
multiplier = tf.constant(2.5, name='Multiplier')
init_zero = tf.constant(0, name='Zero')

condition = lambda i, x: (i < num_iterations)
body = lambda i, x: (i+loop_step, x*multiplier)

i, x = tf.while_loop(condition, body, (init_zero, value_init))

with tf.summary.FileWriter('./log_dir/while') as writer:
    writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(x))


