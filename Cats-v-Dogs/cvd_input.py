import tensorflow as tf
from os import listdir
from os.path import isfile, join, isdir
import re





def read_images(path, batch_size, image_size, augment_data=True, submission_set=False):
    assert isdir(path) is True, "Data dir %r does not exist" % path
    file_list = [path+f for f in listdir(path) if isfile(join(path, f))]
    label_list = []
    id_list = []
    for file in file_list:
        if( "dog" in file):
            label_list.append(1)
        else:
            label_list.append(0)
        regex_result = re.compile('(\d+?)\.', re.S).search(file)
        id_list.append(int(regex_result.group(1)))



    images = tf.convert_to_tensor(file_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    ids = tf.convert_to_tensor(id_list, dtype=tf.int32)

    ## I still have no idea how this input producer thing works, I just read the tutorial  ¯\_(ツ)_/¯
    filename_queue = tf.train.slice_input_producer([images,labels,ids])
    label = filename_queue[1]
    id = filename_queue[2]

    file_contents = tf.read_file(filename_queue[0])
    decoded_img = tf.image.decode_jpeg(file_contents, channels=3)

    ###########################################################
    ################## Image pre processing ###################
    ###########################################################

    if( augment_data is True):
        resized_img = tf.image.resize_images(decoded_img, [image_size,image_size])
        resized_img = tf.random_crop(resized_img, [image_size-10, image_size-10, 3])
        resized_img = tf.image.random_brightness(resized_img,max_delta=63)
        resized_img = tf.image.random_contrast(resized_img,lower=0.2, upper=1.8)
        resized_img = tf.image.random_flip_left_right(resized_img)
        #resized_img = tf.image.random_flip_up_down(resized_img)
    else:
        resized_img = tf.image.resize_images(decoded_img, [image_size-10, image_size-10])

    resized_img = tf.image.per_image_standardization(resized_img)

    # Depends on the number of files and the training speed.
    min_queue_examples = batch_size * 100
    if(submission_set is False):


        images_batch, labels_batch, id_batch = tf.train.shuffle_batch([resized_img,label,id],
                                              batch_size=batch_size,
                                              capacity=min_queue_examples + 3 * batch_size,
                                              min_after_dequeue=min_queue_examples)


    else:
        images_batch, labels_batch, id_batch = tf.train.batch([resized_img, label, id],
                                                                      batch_size=batch_size,
                                                                      capacity=min_queue_examples + 3 * batch_size)

    return images_batch, labels_batch, id_batch