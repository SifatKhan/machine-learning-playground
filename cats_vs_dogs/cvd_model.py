import tensorflow as tf
import numpy as np
from PIL import Image
import math
import time

import cats_vs_dogs.cvd_input as cvd_input


class CVDModel:
    def __init__(self, img_size):

        self._trainingset_size = 23000
        self._batch_size = 128
        self._iterations_per_epoch = int(math.ceil(self._trainingset_size / self._batch_size))

        ## We are going to resize the image, and then randomly crop out 15 pixels.
        self._img_size = img_size
        self._cropped_img_size = self._img_size - 15

        self._x = tf.placeholder(tf.float32, [None, self._cropped_img_size, self._cropped_img_size, 3],
                                 name='Input')
        self._y = tf.placeholder(tf.int32, [None], name="Labels")

        self._weight_decay = tf.placeholder(tf.float32, [], name="WeightDecay")
        ## Keep Probability for Dropout layers.
        self._keep_prob = tf.placeholder(tf.float32, [], name="KeepProbability")

        self.variables = []
        self._training_summaries = []
        self._validation_summaries = []

        self._build_model()
        self._build_trainer()

    def _weight_variable(self, shape, name):
        var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        ## Regularize the weigths to reduce overfitting.
        weight_loss = tf.multiply(tf.nn.l2_loss(var), self._weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
        self.variables.append(var)
        return var

    def _bias_variable(self, shape, name):
        my_var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variables.append(my_var)
        return my_var

    def _build_model(self):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        ## First Convolutional Layer
        with tf.variable_scope('Conv1'):
            W_conv1 = self._weight_variable([4, 4, 3, 64], "W_conv1")
            b_conv1 = self._bias_variable([64], "b_conv1")
            h_conv1 = tf.nn.relu(conv2d(self._x, W_conv1) + b_conv1)

        with tf.variable_scope('Pool1'):
            h1_pool = max_pool_2x2(h_conv1)

        ## Local Response Normalization
        with tf.variable_scope('Norm1'):
            norm1 = tf.nn.lrn(h1_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        ## Second Conv Layer
        with tf.variable_scope('Conv2'):
            W_conv2 = self._weight_variable([3, 3, 64, 100], "W_conv2")
            b_conv2 = self._bias_variable([100], "b_conv2")
            h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2, name="h_conv2")

        with tf.variable_scope('Norm2'):
            norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        with tf.variable_scope('Pool2'):
            h2_pool = max_pool_2x2(norm2)

        ## First Fully-connected layer
        ## We have 2 max pooling layers, so the final Conv layer will be 1/4th the size.
        crop_pool_img_size = int(self._cropped_img_size / 4)
        with tf.variable_scope('FConn1'):
            W_fc1 = self._weight_variable([crop_pool_img_size * crop_pool_img_size * 100, 100], name="W_fc1")
            b_fc1 = self._bias_variable([100], name="b_fc1")
            h2_pool_flat = tf.reshape(h2_pool, [-1, crop_pool_img_size * crop_pool_img_size * 100])
            h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)

        ## Dropout layer to reduce overfitting
        with tf.variable_scope('Dropout1'):
            h_fc1_dropout = tf.nn.dropout(h_fc1, self._keep_prob)

        ## Eliminated the second Fully-connected layer, otherwise the model will be too large
        ## and will exceed the Heroku app's RAM limit.

        with tf.variable_scope('Output'):
            W_output = self._weight_variable([100, 2], "W_output")
            b_output = self._bias_variable([2], "b_output")

            self._result = tf.matmul(h_fc1_dropout, W_output) + b_output
            self.prediction = tf.cast(tf.argmax(self._result, 1), tf.int32)
            self.softmax = tf.nn.softmax(self._result)

        return

    def _build_trainer(self):
        with tf.variable_scope('Train'):
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(None,self._y, self._result))
            tf.add_to_collection('losses', cross_entropy)

            ## Add sum the cross entroy loss plus all the weight l2 losses.
            self._cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
            self._training_summaries.append(tf.summary.scalar('Training Cost', self._cost))
            self._validation_summaries.append(tf.summary.scalar('Validation Cost', self._cost))

            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self._cost)

            correct_prediction = tf.equal(self.prediction, self._y)
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._validation_summaries.append(tf.summary.scalar('accuracy', self._accuracy))

            ## Measure the F1-Score
            true_positives = tf.reduce_sum(
                tf.cast(tf.logical_and(tf.equal(self.prediction, 1), tf.equal(self._y, 1)), tf.int32))
            false_positives = tf.reduce_sum(
                tf.cast(tf.logical_and(tf.equal(self.prediction, 1), tf.equal(self._y, 0)), tf.int32))
            false_negatives = tf.reduce_sum(
                tf.cast(tf.logical_and(tf.equal(self.prediction, 0), tf.equal(self._y, 1)), tf.int32))
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            self._f1_score = 2 * ((precision * recall) / (precision + recall))

            self._validation_summaries.append(tf.summary.scalar('F1-Score', self._f1_score))
            self._validation_summaries.append(tf.summary.scalar('Precision', precision))
            self._validation_summaries.append(tf.summary.scalar('Recall', recall))

        return

    def predict_from_file(self, image_filename, session):
        pic = Image.open(image_filename)
        data = np.asarray(pic, dtype="int32")

        x = tf.placeholder(tf.float32, [None, None, 3])
        resized_img = tf.image.resize_images(x, [self._cropped_img_size, self._cropped_img_size])
        resized_img = tf.image.per_image_standardization(resized_img)
        image = session.run(resized_img, feed_dict={x: data})
        return self.predict([image], session)

    def predict(self, images, session):
        return session.run(self.prediction, feed_dict={self._x: images, self._keep_prob: 1.0})

    def predict_softmax(self, images, session):
        return session.run(self.softmax, feed_dict={self._x: images, self._keep_prob: 1.0})

    def train(self, num_iterations, session):

        with tf.variable_scope('TrainingSet'):
            self._images, self._labels, _ = cvd_input.training_images(self._img_size)

        with tf.variable_scope('ValidationSet'):
            self._val_images, self._val_labels, _ = cvd_input.validation_images(self._img_size)

        training_summary = tf.summary.merge(self._training_summaries)
        validation_summary = tf.summary.merge(self._validation_summaries)
        now = time.time()
        log_dir = '/tmp/cvd_logs/{}'.format(str(int(now)))
        train_writer = tf.summary.FileWriter(log_dir, session.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)

        iter = 0
        while (iter < num_iterations):
            iter += 1
            img_batch, label_batch = session.run([self._images, self._labels])
            _, train_cost, training_summ = session.run([self.train_step, self._cost, training_summary],
                                                       feed_dict={self._x: img_batch, self._y: label_batch,
                                                                  self._keep_prob: 0.5, self._weight_decay: 1e-3})

            if (iter % 10 == 0):
                ## Every 10 iterations, run the model over the validation set to see how we are doing...
                val_img_batch, val_label_batch = session.run([self._val_images, self._val_labels])
                val_accuracy, f1, val_cost, summ = session.run(
                    [self._accuracy, self._f1_score, self._cost, validation_summary],
                    feed_dict={self._x: val_img_batch,
                               self._y: val_label_batch,
                               self._weight_decay: 0.0,
                               self._keep_prob: 1.0})

                train_writer.add_summary(summ, iter)
                train_writer.add_summary(training_summ, iter)

                print("Iter: {0} - Training Loss: {1:.4f} ".format(iter, train_cost), end='')
                print("Validation Loss: {0:.4f} ".format(val_cost), end='')
                print("Validation Accuracy: {0:.4f}, F1 score: {1:.4f}".format(val_accuracy, f1))

            if (iter % self._iterations_per_epoch == 0):
                print("****************End of Epoch {}**************".format(int(iter / self._iterations_per_epoch)))
                tf.train.Saver(self.variables).save(session, 'model/my-model')

        coord.request_stop()
        coord.join(threads)