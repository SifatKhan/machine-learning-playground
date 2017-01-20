import tensorflow as tf
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

import cvd_input

TRAIN_PATH = './train-mini/'
TEST_PATH = './test/'
BATCH_SIZE = 10

images_batch, labels_batch = cvd_input.read_images(TRAIN_PATH,BATCH_SIZE)

xPlaceholder = tf.placeholder(tf.float32, [None, None,None,3],name='xPlaceHolder')

img_op = tf.summary.image("my_img", xPlaceholder,10)

init_op = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init_op)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/cvd_logs/',
                                       session.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    img_batch = session.run(images_batch)
    summary = session.run(merged, feed_dict={xPlaceholder: img_batch})
    train_writer.add_summary(summary)


    coord.request_stop()
    coord.join(threads)

