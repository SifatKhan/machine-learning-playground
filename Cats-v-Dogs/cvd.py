import tensorflow as tf
import cvd_model
from PIL import Image
import numpy as np

MODEL_DIR = 'model/'
TRAIN_DIR = 'train/'
VALIDATION_DIR = 'validate/'
TEST_DIR = 'test/'

model = cvd_model.CVD_Model(model_dir=MODEL_DIR, traindata_dir=TRAIN_DIR,
                            validationdata_dir=VALIDATION_DIR, testdata_dir=TEST_DIR)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model.variables)

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    model.train(5000, session)

    pic = Image.open("picture.jpg")
    data = np.asarray(np.asarray(pic, dtype="int32"))

    x = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = tf.image.resize_images(x, [80, 80])
    resized_img = tf.image.per_image_standardization(resized_img)

    image = session.run(resized_img, feed_dict={x: data})

    prediction = model.predict([image], session)
    if (prediction == 0):
        print("It's a CAT!")
    else:
        print("It's a DOG!")
