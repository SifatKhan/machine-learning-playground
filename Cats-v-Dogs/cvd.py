import tensorflow as tf
import time

import cvd_input

now = time.time()

TRAIN_PATH = './train/'
TEST_PATH = './validate/'
BATCH_SIZE = 100
ITERATIONS = 6000
IMAGE_SIZE = 90

cropped_and_pool_img_size = int((IMAGE_SIZE-10)/4)
iterations_per_epoch = 23000 / BATCH_SIZE

with tf.variable_scope('TrainingSet'):
    images, labels = cvd_input.read_training_images(TRAIN_PATH, BATCH_SIZE,IMAGE_SIZE)

with tf.variable_scope('ValidationSet'):
    test_images, test_labels = cvd_input.read_training_images(TEST_PATH, BATCH_SIZE,IMAGE_SIZE,augment_data=False)

x_placeholder = tf.placeholder(tf.float32, [None,IMAGE_SIZE-10,IMAGE_SIZE-10,3],name='input')
y_placeholder = tf.placeholder(tf.int32, [None],name="labels")
keep_prob = tf.placeholder(tf.float32, name="KeepProbability")
img_op = tf.summary.image("my_img", x_placeholder,50)


def weight_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


with tf.variable_scope('Conv1'):
    W_conv1 = weight_variable([8,8,3,64],"W_conv1")
    b_conv1 = bias_variable([64],"b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_placeholder,W_conv1)+b_conv1)

with tf.variable_scope('Pool1'):
    h1_pool = max_pool_2x2(h_conv1)

with tf.variable_scope('Norm1'):
    norm1 = tf.nn.lrn(h1_pool, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

with tf.variable_scope('Conv2'):
    W_conv2 = weight_variable([8,8,64,128],"W_conv2")
    b_conv2 = bias_variable([128],"b_conv2")
    h_conv2 = tf.nn.relu(conv2d(norm1,W_conv2)+b_conv2)

with tf.variable_scope('Norm2'):
    norm2 = tf.nn.lrn(h_conv2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

with tf.variable_scope('Pool2'):
    h2_pool = max_pool_2x2(norm2)

with tf.variable_scope('FConn1'):
    W_fc1 = weight_variable([cropped_and_pool_img_size*cropped_and_pool_img_size*128,1024],name="W_fc")
    b_fc1 = bias_variable([1024],name="b_fc")
    h2_pool_flat = tf.reshape(h2_pool,[-1,cropped_and_pool_img_size*cropped_and_pool_img_size*128])
    h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat,W_fc1)+b_fc1)

with tf.variable_scope('Dropout'):
    h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

with tf.variable_scope('Output'):
    W_output = weight_variable([1024,2],"W_output")
    b_output = bias_variable([2],"b_output")

    prediction = tf.matmul(h_fc1_dropout,W_output)+b_output
    prediction_argmax = tf.cast(tf.argmax(prediction, 1),tf.int32)

with tf.variable_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction,y_placeholder))
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(prediction_argmax, y_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,1),tf.equal(y_placeholder, 1) ),tf.int32))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,1),tf.equal(y_placeholder, 0) ),tf.int32))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,0),tf.equal(y_placeholder, 1) ),tf.int32))
    precision =  true_positives / (true_positives +false_positives )
    recall = true_positives / (true_positives +false_negatives )
    f1_score = 2 * ((precision * recall) / (precision + recall))

    tf.summary.scalar('true_positives', true_positives)
    tf.summary.scalar('false_positives', false_positives)
    tf.summary.scalar('false_negatives', false_negatives)
    tf.summary.scalar('a-F1-Score', f1_score)

iter = 0

init_op = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()

    log_dir = '/tmp/cvd_logs/{}'.format( str(int(now)))
    train_writer = tf.summary.FileWriter(log_dir, session.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while( iter < ITERATIONS ):
        iter+=1
        img_batch, label_batch = session.run([images,labels])
        _,_loss,_pred = session.run([train_step,cross_entropy,prediction],
                              feed_dict={x_placeholder: img_batch, y_placeholder: label_batch,
                                         keep_prob: 0.5})


        if( iter > 20 and iter % 10 == 0 ):


            val_img_batch, val_label_batch = session.run([test_images, test_labels])
            accuracy_result,_test_pred_argmax,_f1,summary = session.run([accuracy, prediction_argmax,f1_score,merged],
                                                                         feed_dict={x_placeholder: val_img_batch,
                                                                                    y_placeholder: val_label_batch,
                                                                                    keep_prob: 1.0})
            train_writer.add_summary(summary, iter)


            if( iter+1 % iterations_per_epoch == 0 ):
                print("*********************************************")
                print("****************End of Epoch {}**************".format(int(iter/iterations_per_epoch)))
                print("*********************************************")

            print("Iteration: {0}, Loss: {1:.4f} Validation accuracy: {2:.4f}, F1 score: {3:.4f}".format( iter,_loss,accuracy_result,_f1))

    coord.request_stop()
    coord.join(threads)

