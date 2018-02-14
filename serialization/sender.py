import tensorflow as tf

builder = tf.saved_model.builder.SavedModelBuilder('./export')

######Build graph######
a = tf.get_variable('var_a', initializer=tf.zeros(shape=[5]))
b = tf.get_variable('var_b', initializer=tf.zeros(shape=[5]))
c = tf.add(a, b, name='c')

dec_a = a.assign(a - 1)
inc_b = b.assign(b + 1)
train_step = tf.train.AdamOptimizer().minimize(c)
######################

init_op = tf.global_variables_initializer()

####PRETTY INTERESTING FEATURE####
tf.add_to_collection('learning_rate', 0.33)
#################################

with tf.Session() as sess:
    sess.run(init_op)
    sess.run([dec_a, inc_b])
    
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=['foo-tag', 'bar-tag'])
                                                            
builder.save(as_text=True)

