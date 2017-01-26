import tensorflow as tf
import numpy as np
import pandas as pd
import cvd_input
import cvd_model
from os.path import exists

'''
Script to generate the submission file for the Kaggle Cats-vs-Dogs dataset.
'''

MODEL_DIR = 'model/'
if not exists(MODEL_DIR):
    print("Did not find a trained model.")
    exit(1)

model = model = cvd_model.CVDModel(img_size=115)
test_images, _, test_ids = cvd_input.test_images(image_size=115)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model.variables)

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Did not find a trained model.")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord=coord)

    iter = 0
    myPrediction = np.array([])
    myIds = np.array([])
    while (iter < int(12500 / 100)):  #

        iter += 1

        if (iter % 5 == 0): print("Evaluation iteration {}".format(iter))

        test_img_batch, test_id_batch = session.run([test_images, test_ids])
        prediction = model.predict_softmax(test_img_batch,session)

        myPrediction = np.hstack((prediction[:, 1], myPrediction))
        myIds = np.hstack((test_id_batch.astype(int), myIds))


    ## Write the submission data into a CSV file.
    data = np.vstack((myIds, myPrediction)).T

    df = pd.DataFrame(data=data, columns=['id', 'label'])
    df.label = df.label.astype(float)
    df.id = df.id.astype(int)
    df.to_csv('output.csv', index=False)

    coord.request_stop()
    coord.join(threads)
