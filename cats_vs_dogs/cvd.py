import tensorflow as tf
import time

import cats_vs_dogs.cvd_input as cvd_input


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def _weight_variable(shape, name):
    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return var

def _bias_variable(shape, name):
    my_var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return my_var

def inference(images):
    ## First Convolutional Layer
    with tf.variable_scope('Conv1'):
        W_conv1 = _weight_variable([4, 4, 3, 96], "W_conv1")
        b_conv1 = _bias_variable([96], "b_conv1")
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)

    with tf.variable_scope('Pool1'):
        h1_pool = max_pool_2x2(h_conv1)

    ## Local Response Normalization
    with tf.variable_scope('Norm1'):
        norm1 = tf.nn.lrn(h1_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    ## Second Conv Layer
    with tf.variable_scope('Conv2'):
        W_conv2 = _weight_variable([3, 3, 96, 192], "W_conv2")
        b_conv2 = _bias_variable([192], "b_conv2")
        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2, name="h_conv2")

    with tf.variable_scope('Norm2'):
        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.variable_scope('Pool2'):
        h2_pool = max_pool_2x2(norm2)

    ## First Fully-connected layer
    ## We have 2 max pooling layers, so the final Conv layer will be 1/4th the size.
    crop_pool_img_size = int(100 / 4)
    with tf.variable_scope('FConn1'):
        W_fc1 = _weight_variable([crop_pool_img_size * crop_pool_img_size * 192, 32], name="W_fc1")
        b_fc1 = _bias_variable([32], name="b_fc1")
        h2_pool_flat = tf.reshape(h2_pool, [-1, crop_pool_img_size * crop_pool_img_size * 192])
        h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)

    ## Dropout layer to reduce overfitting
    with tf.variable_scope('Dropout1'):
        h_fc1_dropout = tf.nn.dropout(h_fc1, 1.0)

    ## Second Fully-connected layer
    with tf.variable_scope('FConn2'):
        W_fc2 = _weight_variable([32, 32], name="W_fc2")
        b_fc2 = _bias_variable([32], name="b_fc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

    ## Second dropout layer
    with tf.variable_scope('Dropout2'):
        h_fc2_dropout = tf.nn.dropout(h_fc2, 1.0)

    with tf.variable_scope('Output'):
        W_output = _weight_variable([32, 2], "W_output")
        b_output = _bias_variable([2], "b_output")

        softmax_linear = tf.matmul(h_fc2_dropout, W_output) + b_output

    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    entropy_mean = tf.reduce_mean(cross_entropy)
    return entropy_mean


def train(total_loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    return train_step



training_images, training_labels, _ = cvd_input.training_images(115)
softmax_linear = inference(training_images)
cost = loss(softmax_linear,training_labels)
train_step = train(cost)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord=coord)

    iter = 0
    while (iter < 10000):
        start_time = time.time()
        iter+=1
        _, train_cost = session.run([train_step,cost])
        current_time = time.time()
        duration = current_time - start_time
        start_time = current_time

        print("Iter: {0} - Training Loss: {1:.4f} {2:.3f} sec/batch".format(iter, train_cost, duration))

    coord.request_stop()
    coord.join(threads)
