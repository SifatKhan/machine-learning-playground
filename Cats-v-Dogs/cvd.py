import tensorflow as tf
import cvd_model
from PIL import Image
import numpy as np
from os.path import exists
from os import makedirs

MODEL_DIR = 'model/'
IMAGE_SIZE = 115
if not exists(MODEL_DIR): makedirs(MODEL_DIR)


model = cvd_model.CVDModel(img_size=IMAGE_SIZE)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model.variables)

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    model.train(35000, session)

    pic = Image.open("picture.jpg")
    data = np.asarray(np.asarray(pic, dtype="int32"))

    x = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = tf.image.resize_images(x, [100, 100])
    resized_img = tf.image.per_image_standardization(resized_img)

    image = session.run(resized_img, feed_dict={x: data})

    prediction = model.predict([image], session)
    if (prediction == 0):
        print("It's a CAT!")
    else:
        print("It's a DOG!")
