import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def sigmoid_op(inputs):
    
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def _sigmoid_derivative(x):
        return _sigmoid(x) * (1 - _sigmoid(x))

    def _sigmoid_der_op(op, grad):
        x = op.inputs[0]
        x_der = grad * tf.py_func(_sigmoid_derivative, [x], tf.float32)
        return x_der

    tf.RegisterGradient('YetAnotherSigmoidGrad')(_sigmoid_der_op)

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": 'YetAnotherSigmoidGrad'}):
        result = tf.py_func(_sigmoid, [inputs], tf.float32)
    return result

x = tf.random_normal([100])
y = sigmoid_op(x)

with tf.summary.FileWriter('./tb') as writer:
    writer.add_graph(tf.get_default_graph())

with tf.Session():
    error = tf.test.compute_gradient_error(x, [100], y, [100])
    print(error)


