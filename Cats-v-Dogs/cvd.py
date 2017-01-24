import tensorflow as tf
import pandas as pd
import numpy as np
import cvd_model
import cvd_input

from os import listdir
from os.path import isfile, join, isdir

MODEL_DIR = './model/'
TRAIN_DIR = './train/'
VALIDATION_DIR = './validate/'
TEST_DIR = './test/'
IMAGE_SIZE = 90

testset_size = len([f for f in listdir(TEST_DIR) if isfile(join(TEST_DIR, f))])
test_images, _, test_ids = cvd_input.read_training_images(TEST_DIR, 100, IMAGE_SIZE, augment_data=False,
                                                          submission_set=True)

model = cvd_model.CVD_Model(model_dir=MODEL_DIR, traindata_dir=TRAIN_DIR,
                            validationdata_dir=VALIDATION_DIR)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model.variables)

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("*********************************************")
    print("**************Finished training**************")
    print("*********************************************")

    iter = 0
    myPrediction = None
    myIds = None
    while (iter < int(testset_size / 100)):  #

        if (iter % 5 == 0):
            print("Evaluation iteration {}".format(iter))

        iter += 1

        test_img_batch, test_id_batch = session.run([test_images, test_ids])
        _test_pred = model.predict_softmax(test_img_batch, session)

        if (myPrediction is None):
            myPrediction = np.asmatrix(_test_pred[:, 1]).T
            myIds = np.asmatrix(test_id_batch.astype(int)).T
        else:
            myPrediction = np.vstack((myPrediction, np.asmatrix(_test_pred[:, 1]).T))
            myIds = np.vstack((myIds, np.asmatrix(test_id_batch.astype(int)).T))

    data = np.hstack((myIds, myPrediction))

    df = pd.DataFrame(data=data, columns=['id', 'label'])
    df.label = df.label.astype(float)
    df.id = df.id.astype(int)
    df.to_csv('output.csv', index=False)

    coord.request_stop()
    coord.join(threads)
