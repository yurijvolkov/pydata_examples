import tensorflow as tf

def weight_init(name, shape):
    return tf.get_variable(name, initializer=tf.random_normal(shape=shape,
                                                              stddev=0.1))
def bias_init(name, shape):
  return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))

#### Building Graph ####
NUM_LAYERS = 1
DEPTH = 20
STATE_SIZE = 500
NUM_EPOCHS = 200

x = tf.placeholder(tf.float32, shape=[None, None, DEPTH], name='x')
y = tf.placeholder(tf.float32, shape=[None, None, DEPTH], name='y')
batch_size = tf.placeholder(tf.int32, shape=[None], name='batch_size')
dropout_rate = tf.placeholder(tf.float32, shape=[None], name='dropout_rate')

stacked_rnn = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(STATE_SIZE),
                                             output_keep_prob=dropout_rate[0]) 
                for _ in range(NUM_LAYERS)]

cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn)
init_state = cell.zero_state(batch_size[0], tf.float32)

preds, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)

w = weight_init('Dense_1_Weights', [STATE_SIZE, DEPTH])
b = bias_init('Dense_1_Bias', [DEPTH])

logits = tf.map_fn(lambda x: tf.matmul(x, w) + b, preds, name='logits')

loss = tf.reduce_mean(tf.abs(y - logits), name='loss')

train_step = tf.train.AdamOptimizer().minimize(loss, name='train_step')


#### Logging graph definition ####

with tf.summary.FileWriter('./log_dir/tb') as writer:
    writer.add_graph(tf.get_default_graph())



