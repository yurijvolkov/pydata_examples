import tensorflow as tf

#### Value stored in tf.constant ####
print('With constant :')
foo_const = tf.constant([0., 0.], name='foo_const')
print(tf.get_default_graph().as_graph_def())


tf.reset_default_graph()

#### Value stored in tf.Variable ####
print('\n\nWith variable :')
bar_var = tf.Variable([0., 0.], name='bar_var')
print(tf.get_default_graph().as_graph_def())


tf.reset_default_graph()

#### Value not created in compilation time ####
print('\n\nWith lazy initializtion :')
initializer = tf.zeros([2], tf.float32)
foo_var = tf.get_variable('foo_var', initializer=initializer)
print(tf.get_default_graph().as_graph_def())
