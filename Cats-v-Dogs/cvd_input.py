import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join, isdir





def read_images(path,batch_size):
    assert isdir(path) is True, "Data dir %r does not exist" % path
    file_list = [path+f for f in listdir(path) if isfile(join(path, f))]
    label_list = []
    for file in file_list:
        if( "dog" in file):
            label_list.append(1)
        else:
            label_list.append(0)

    images = tf.convert_to_tensor(file_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    filename_queue = tf.train.slice_input_producer([images,labels])

    file_contents = tf.read_file(filename_queue[0])
    my_img = tf.image.decode_jpeg(file_contents, channels=3)
    resized_img = tf.image.resize_images(my_img, [80,80])
    label = filename_queue[1]


    # Depends on the number of files and the training speed.
    min_queue_examples = batch_size * 100
    images_batch, labels_batch = tf.train.shuffle_batch([resized_img,label],
                                          batch_size=batch_size,
                                          capacity=min_queue_examples + 3 * batch_size,
                                          min_after_dequeue=min_queue_examples)

    return images_batch, labels_batch