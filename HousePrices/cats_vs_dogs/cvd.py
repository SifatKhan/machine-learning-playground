from os import makedirs
from os.path import exists

import tensorflow as tf

import HousePrices.cats_vs_dogs.cvd_model as cvd_model

MODEL_DIR = 'model/'
IMAGE_SIZE = 95
if not exists(MODEL_DIR): makedirs(MODEL_DIR)
model = cvd_model.CVDModel(img_size=IMAGE_SIZE)


def cat_or_dog(file):
    prediction = model.predict_from_file(file, session)
    if (prediction == 0):
        print("{0} is a CAT!".format(file))
    else:
        print("{0} a DOG!".format(file))


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model.variables)

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        model.train(num_iterations=20000, session=session)

    cat_or_dog("picture.jpg")
    cat_or_dog("dog.jpg")
