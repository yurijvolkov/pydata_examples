import tensorflow as tf
import argparse
import logging

logging.getLogger().setLevel(logging.ERROR)

TF_GRAPH_PATH = './export'

def restore(sess):
    """
        Deserializing model.
    
    :param sess: tf.Session

    :return: list
    """

    tf.saved_model.loader.load(sess, ['foo-tag', 'bar-tag'], TF_GRAPH_PATH)
    graph = tf.get_default_graph()
    a = graph.get_tensor_by_name('var_a:0')
    b = graph.get_tensor_by_name('var_b:0')
    c = graph.get_tensor_by_name('c:0')
    return [a,b,c]

with tf.Session() as sess:
    a,b,c = restore(sess)
    print(tf.get_collection('learning_rate'))

    print(sess.run([b]))
    print(sess.run([c], feed_dict={a: [1,1,1,1,1]}))
