import tensorflow as tf
import time
import pandas as pd
import numpy as np
import os.path

import cvd_input

now = time.time()

MODEL_DIR = './model/'
TRAIN_PATH = './train/'
VALIDATION_PATH = './validate/'
TEST_PATH = './test/'
TEST_SET_SIZE = 12500
TRAINING_SET_SIZE = 23000
BATCH_SIZE = 128
ITERATIONS = 50000
IMAGE_SIZE = 90

cropped_and_pool_img_size = int((IMAGE_SIZE-10)/4)
iterations_per_epoch = int(TRAINING_SET_SIZE / BATCH_SIZE)+1

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with tf.variable_scope('TrainingSet'):
    images, labels,ids = cvd_input.read_training_images(TRAIN_PATH, BATCH_SIZE,IMAGE_SIZE,augment_data=True)

with tf.variable_scope('ValidationSet'):
    val_images, val_labels, val_ids = cvd_input.read_training_images(VALIDATION_PATH, 500, IMAGE_SIZE, augment_data=False)

with tf.variable_scope('TestSet'):
    test_images, _, test_ids = cvd_input.read_training_images(TEST_PATH, 100, IMAGE_SIZE, augment_data=False, submission_set=True)

x_placeholder = tf.placeholder(tf.float32, [None,IMAGE_SIZE-10,IMAGE_SIZE-10,3],name='Input')
y_placeholder = tf.placeholder(tf.int32, [None],name="Labels")
weight_decay_placeholder = tf.placeholder(tf.float32,[],name="WeightDecay")
keep_prob = tf.placeholder(tf.float32, name="KeepProbability")
img_op = tf.summary.image("my_img", x_placeholder,50)


def weight_variable(shape,name):
    my_var = tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('vars', my_var)
    weight_decay = tf.mul(tf.nn.l2_loss(my_var), weight_decay_placeholder, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return my_var

def bias_variable(shape,name):
    my_var =  tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('vars', my_var)
    return my_var

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
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
    W_conv2 = weight_variable([4,4,64,128],"W_conv2")
    b_conv2 = bias_variable([128],"b_conv2")
    h_conv2 = tf.nn.relu(conv2d(norm1,W_conv2)+b_conv2)

with tf.variable_scope('Norm2'):
    norm2 = tf.nn.lrn(h_conv2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

with tf.variable_scope('Pool2'):
    h2_pool = max_pool_2x2(norm2)

with tf.variable_scope('FConn1'):
    W_fc1 = weight_variable([cropped_and_pool_img_size*cropped_and_pool_img_size*128,1024],name="W_fc1")
    b_fc1 = bias_variable([1024],name="b_fc1")
    h2_pool_flat = tf.reshape(h2_pool,[-1,cropped_and_pool_img_size*cropped_and_pool_img_size*128])
    h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat,W_fc1)+b_fc1)

with tf.variable_scope('Dropout1'):
    h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)


with tf.variable_scope('FConn2'):
    W_fc2 = weight_variable([1024,1024],name="W_fc2")
    b_fc2 = bias_variable([1024],name="b_fc2")
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout,W_fc2)+b_fc2)

with tf.variable_scope('Dropout2'):
    h_fc2_dropout = tf.nn.dropout(h_fc2,keep_prob)

with tf.variable_scope('Output'):
    W_output = weight_variable([1024,2],"W_output")
    b_output = bias_variable([2],"b_output")

    prediction = tf.matmul(h_fc2_dropout,W_output)+b_output
    prediction_argmax = tf.cast(tf.argmax(prediction, 1),tf.int32)
    softmax = tf.nn.softmax(prediction)

with tf.variable_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction,y_placeholder))
    tf.add_to_collection('losses', cross_entropy)
    cost = tf.add_n(tf.get_collection('losses'), name='total_loss')

    cost_op = tf.summary.scalar('cost', cost)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    correct_prediction = tf.equal(prediction_argmax, y_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_op = tf.summary.scalar('accuracy', accuracy)
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,1),tf.equal(y_placeholder, 1) ),tf.int32))
    true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax, 0), tf.equal(y_placeholder, 0)), tf.int32))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,1),tf.equal(y_placeholder, 0) ),tf.int32))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction_argmax,0),tf.equal(y_placeholder, 1) ),tf.int32))
    precision =  true_positives / (true_positives +false_positives )
    recall = true_positives / (true_positives +false_negatives )
    f1_score = 2 * ((precision * recall) / (precision + recall))


    f1_scope_op = tf.summary.scalar('F1-Score', f1_score)
    recall_op = tf.summary.scalar('Recall', recall)
    precision_op = tf.summary.scalar('Precision', precision)


iter = 0

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    merged = tf.summary.merge(
        [f1_scope_op,recall_op, precision_op, accuracy_op, cost_op])

    log_dir = '/tmp/cvd_logs/{}'.format( str(int(now)))
    train_writer = tf.summary.FileWriter(log_dir, session.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while( iter < ITERATIONS ):
        iter+=1
        img_batch, label_batch = session.run([images,labels])
        _,_pred = session.run([train_step,prediction],
                              feed_dict={x_placeholder: img_batch, y_placeholder: label_batch,
                                         keep_prob: 0.5, weight_decay_placeholder: 5e-4})

        if( iter % 10 == 0 ):


            val_img_batch, val_label_batch, val_id_batch = session.run([val_images, val_labels,val_ids])
            accuracy_result,_f1,_loss,summary = session.run([accuracy,f1_score,cost,merged],
                                                     feed_dict={x_placeholder: val_img_batch,
                                                                y_placeholder: val_label_batch,
                                                                weight_decay_placeholder: 0.0,
                                                                keep_prob: 1.0})
            train_writer.add_summary(summary, iter)


            print("Iteration: {0}, Validation Loss: {1:.4f} ".format( iter,_loss),end='')
            print("Validation accuracy: {0:.4f}, F1 score: {1:.4f}".format(accuracy_result, _f1))

        if (iter % iterations_per_epoch == 0):
            print("*********************************************")
            print("****************End of Epoch {}**************".format(int(iter / iterations_per_epoch)))
            print("*********************************************")

            saver.save(session, MODEL_DIR+'my-model')

    print("*********************************************")
    print("**************Finished training**************")
    print("*********************************************")

    iter = 0
    myPrediction = None
    myIds = None
    while (iter < int(TEST_SET_SIZE/100)): #

        if (iter % 5 == 0):
            print("Evaluation iteration {}".format(iter))

        iter += 1

        test_img_batch, test_id_batch = session.run([test_images, test_ids])
        _test_pred  = session.run(softmax,feed_dict={x_placeholder: test_img_batch,
                                                                    keep_prob: 1.0})

        if( myPrediction is None ):
            myPrediction =  np.asmatrix(_test_pred[:, 1]).T
            myIds = np.asmatrix(test_id_batch.astype(int)).T
        else:
            myPrediction = np.vstack((myPrediction,np.asmatrix(_test_pred[:,1]).T))
            myIds = np.vstack((myIds, np.asmatrix(test_id_batch.astype(int)).T))



    data = np.hstack((myIds, myPrediction))

    df = pd.DataFrame(data=data, columns=['id', 'label'])
    df.label = df.label.astype(float)
    df.id = df.id.astype(int)
    df.to_csv('output.csv', index=False)


    coord.request_stop()
    coord.join(threads)

