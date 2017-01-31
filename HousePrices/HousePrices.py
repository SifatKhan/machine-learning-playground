import numpy as np
import tensorflow as tf
import pandas as pd
import houselist as hl
import time

pd.read_csv('data/train.csv').sample(frac=1).to_csv('data/train.csv',index=False)
house_list = hl.HouseList(trainfile='data/train.csv',testfile='data/test.csv',use_validationset=False)

def weight_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

input_nodes = house_list.traindata.shape[1]
hidden_nodes_layer1 = 200
hidden_nodes_layer2 = 200
hidden_nodes_layer3 = 200
output_nodes = 1
num_iterations = 25000


xplaceholder = tf.placeholder(tf.float32, [None, input_nodes])
yplaceholder = tf.placeholder(tf.float32, [None,1])
lambdaPlaceHolder = tf.placeholder(tf.float32, [])


with tf.name_scope('Hidden1'):
    hidden_layer1 = {'theta': weight_variable([input_nodes, hidden_nodes_layer1], 'Theta1'),
                     'bias': bias_variable([hidden_nodes_layer1], 'Bias1')}
    a2 = tf.nn.relu(tf.add(tf.matmul(xplaceholder, hidden_layer1['theta']), hidden_layer1['bias']), name='Layer2Activation')


with tf.name_scope('Hidden2'):
    hidden_layer2 = {'theta': weight_variable([hidden_nodes_layer1, hidden_nodes_layer2], 'Theta2'),
                     'bias': bias_variable([hidden_nodes_layer2], 'Bias2')}
    a3 = tf.nn.relu(tf.add(tf.matmul(a2, hidden_layer2['theta']), hidden_layer2['bias']), name='Layer3Activation')


with tf.name_scope('Hidden3'):
    hidden_layer3 = {'theta': weight_variable([hidden_nodes_layer2, hidden_nodes_layer3], 'Theta3'),
                     'bias': bias_variable([hidden_nodes_layer3], 'Bias3')}
    a4 = tf.nn.relu(tf.add(tf.matmul(a3, hidden_layer3['theta']), hidden_layer3['bias']), name='Layer4Activation')

with tf.name_scope("Dropout"):
    a4_drop = tf.nn.dropout(a4, lambdaPlaceHolder)

with tf.name_scope('OutputLayer'):
    output_layer = {'theta': weight_variable([hidden_nodes_layer3, output_nodes], 'Theta4'),
                    'bias': bias_variable([output_nodes], 'Bias4')}
    prediction = tf.add(tf.matmul(a4_drop, output_layer['theta']), output_layer['bias'],name='Prediction')

with tf.name_scope('CostFunction'):
    cost = tf.sqrt(tf.reduce_mean(tf.square(prediction - yplaceholder)))
    log_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.log(tf.maximum(prediction,1.0)) - tf.log(yplaceholder))))
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



lambdaValues = [1,1,1,1,
                5e-1,5e-1,5e-1,5e-1]

lambdaValues = [5e-1]
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    lambdaIndex = 0
    now = time.time()
    for myLambda in lambdaValues:

        session.run(tf.global_variables_initializer())

        log_dir = '/tmp/house_logs/'+str(int(now))+'-'+str(myLambda)+'['+str(lambdaIndex)+']'
        lambdaIndex+=1

        train_writer = tf.summary.FileWriter(log_dir,session.graph)


        for iteration in range(num_iterations):

            _, temp_cost, summary,_pre, = session.run([optimizer, cost, merged,prediction],
                                                    feed_dict={xplaceholder: house_list.traindata,
                                                               yplaceholder: house_list.trainlabels,
                                                               lambdaPlaceHolder: myLambda})

            if (iteration % 250 == 0):
                train_writer.add_summary(summary, iteration)
                print('Training error:', temp_cost, ' Iteration', iteration + 1, 'out of', num_iterations)


        validationPrediction,validationCost = session.run([prediction,log_cost],
                                                          feed_dict={xplaceholder: house_list.validationdata,
                                                                     yplaceholder:house_list.validationlabels,
                                                                     lambdaPlaceHolder: 1.0})


        print("Validation Cost:",validationCost, "Lambda:",myLambda,sep='\t')
        coord.join(threads)


    testing_x = house_list.testdata
    testingIds= house_list.test_ids
    myPrediction = session.run(prediction, feed_dict={xplaceholder: testing_x, lambdaPlaceHolder:1.0})

    myPrediction = np.asmatrix(myPrediction)
    data = np.hstack((testingIds.astype(int),myPrediction))

    df = pd.DataFrame(data=data,columns = ['Id','SalePrice'])
    df.SalePrice = df.SalePrice.astype(float)
    df.Id = df.Id.astype(int)
    df.to_csv('data/output.csv', index=False)

    coord.request_stop()