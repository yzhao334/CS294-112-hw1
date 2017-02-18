import pickle, tensorflow as tf, tf_util, numpy as np
#from tensorflow.contrib.labeled_tensor import batch

def loss(y_, y):
    cost = tf.nn.l2_loss(tf.sub(y_,y))
    return cost

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trainstep = optimizer.minimize(loss)
    return trainstep    

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_uniform(shape=shape,minval=-np.sqrt(6.0/sum(shape)),
                                maxval=np.sqrt(6.0/sum(shape)))
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.tanh):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses tanh to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

def inference(x, nin, nout, n_h1, n_h2, n_h3):
    layer1 = nn_layer(x, nin, n_h1, 'layer1')
    layer2 = nn_layer(layer1, n_h1, n_h2, 'layer2')
    layer3 = nn_layer(layer2, n_h2, n_h3, 'layer3')
    with tf.name_scope('outlayer'):
        with tf.name_scope('weights'):
            W = weight_variable([n_h3, nout])
        with tf.name_scope('biases'):
            b = bias_variable([nout])
    out = tf.matmul(layer3, W) + b
    #with tf.name_scope('dropout'):
    #    #keep_prob = tf.placeholder(tf.float32)
    #    keep_prob = 0.8
    #    tf.summary.scalar('dropout_keep_probability', keep_prob)
    #    dropped = tf.nn.dropout(out, keep_prob)
    # Do not apply softmax activation yet, see below.
    #y = nn_layer(dropped, nout, nout, 'output',tf.identity)
    return out

def placeholder_inputs(size,nin,nout, batch_size):
  x_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                         nin))
  y_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                         nout))
  return x_placeholder, y_placeholder

def fill_feed_dict(x, y, data, i, nin, nout, batch_size):
    '''
    arr = np.arange(len(data['actions']))
    np.random.shuffle(arr)
    x_feed = data['observations'][arr[i*batch_size:(i+1)*batch_size]]
    y_feed = data['actions'][arr[i*batch_size:(i+1)*batch_size]].reshape(batch_size,nout)
    '''
    x_feed = data['observations'][i*batch_size:(i+1)*batch_size]
    y_feed = data['actions'][i*batch_size:(i+1)*batch_size].reshape(batch_size,nout)
    feed_dict = {
        x: x_feed,
        y: y_feed,
    }
    return feed_dict
