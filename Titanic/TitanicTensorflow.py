import tensorflow as tf
import tensorflow.contrib.slim as tfslim
import pandas as pd
import numpy as np
import time
import passengerlist


#pd.read_csv('train.csv').sample(frac=1).to_csv('train.csv',index=False)
myDf = pd.read_csv('train.csv')
tickets = np.unique(np.append(pd.read_csv('test.csv').Ticket.unique(),myDf.Ticket.unique()))

myDf['Surname'] = myDf['Name'].str.extract('(\w+),\s*?\w+',expand=True)
surnames = myDf.Surname.unique()
myDf = pd.read_csv('test.csv')
myDf['Surname'] = myDf['Name'].str.extract('(\w+),\s*?\w+',expand=True)
surnames = np.unique(np.append(surnames,myDf.Surname.unique()))



test_passenger_list = passengerlist.read_csv('test.csv',tickets,surnames,testlist=True)

## Ok, here we being to train using a neural network

def weight_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


num_iterations = 2500
input_nodes = 22 + len(tickets) + len(surnames) #951 # 22
hidden_nodes_layer1 = 500
hidden_nodes_layer2 = 500
hidden_nodes_layer3 = 500
output_nodes = 2

xplaceholder = tf.placeholder(tf.float32, [None, input_nodes])
yplaceholder = tf.placeholder(tf.int32, [None,1])
lambdaPlaceHolder = tf.placeholder(tf.float32, [])

yplaceholder_onehot = tfslim.one_hot_encoding(yplaceholder,output_nodes)

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
    output_layer = {'theta': weight_variable([hidden_nodes_layer3, output_nodes], 'Theta6'),
                    'bias': bias_variable([output_nodes], 'Bias6')}
    prediction = tf.nn.softmax(tf.add(tf.matmul(a4_drop, output_layer['theta']), output_layer['bias'],name='Prediction'))
    prediction_argmax = tf.argmax(prediction,1)



with tf.name_scope('CostFunction'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, yplaceholder_onehot))

    '''
    cost = tf.reduce_mean(cost + lambdaPlaceHolder * tf.nn.l2_loss(hidden_layer1['theta']) \
                        + lambdaPlaceHolder * tf.nn.l2_loss(hidden_layer2['theta']) \
                        + lambdaPlaceHolder * tf.nn.l2_loss(hidden_layer3['theta']))
    '''
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



passenger_list = passengerlist.read_csv('train.csv',tickets,surnames)

lambdaValues = [
                4e-3,
                4e-3,
                4e-3,
                4e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                2e-3,
                2e-3,
                2e-3,
                2e-3,
               ]

lambdaValues = [1e-3]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    lambdaIndex = 0
    now = time.time()
    for myLambda in lambdaValues:

        session.run(tf.global_variables_initializer())

        log_dir = '/tmp/titanic_logs/'+str(int(now))+'-'+str(myLambda)+'['+str(lambdaIndex)+']'
        lambdaIndex+=1

        train_writer = tf.summary.FileWriter(log_dir,
                                             session.graph)

        batch_x = passenger_list.data[0:]
        batch_y = passenger_list.get_survival_data()[0:]


        val_batch_x = passenger_list.data[-100:]
        val_batch_y = passenger_list.get_survival_data()[-100:]

        for iteration in range(num_iterations):

            _, temp_cost, summary,_pre = session.run([optimizer, cost, merged,prediction_argmax],
                                        feed_dict={xplaceholder: batch_x, yplaceholder: batch_y, lambdaPlaceHolder: 1.0})

            if (iteration % 250 == 0):
                train_writer.add_summary(summary, iteration)
                print('Training error:', temp_cost, ' Iteration', iteration + 1, 'out of', num_iterations)

        testPrediction = session.run(prediction_argmax, {xplaceholder: batch_x, yplaceholder: batch_y, lambdaPlaceHolder: 1.0})
        testPrediction = np.asmatrix(testPrediction).T
        accuracy = np.equal(batch_y, testPrediction)
        accuracyMean = np.mean(accuracy)


        validationPrediction = session.run(prediction_argmax,{xplaceholder: val_batch_x, yplaceholder:val_batch_y, lambdaPlaceHolder: 1.0})
        validationPrediction = np.asmatrix(validationPrediction).T
        accuracy = np.equal(val_batch_y,validationPrediction)
        validationAccuracyMean = np.mean(accuracy)

        temp_matrix = np.hstack([validationPrediction,val_batch_y])

        false_positives = 0
        true_positives = 0
        false_negatives = 0
        for entry in temp_matrix:
            if(entry[0,0] == 1 and entry[0,1] == 0 ): false_positives+=1
            if(entry[0,0] == 1 and entry[0,1] == 1): true_positives += 1
            if(entry[0,0] == 0 and entry[0,1] == 1): false_negatives += 1

        prec = true_positives / (true_positives + false_positives)
        rec = true_positives / (true_positives + false_negatives)

        f1 = 2* ((prec * rec) / (prec + rec) )


        print("Test Accuracy:",accuracyMean,"Validation Acurracy: %.2f" % validationAccuracyMean,"F1 Score:",f1, "Lambda:",myLambda,sep='\t')
        coord.join(threads)


    testing_x = test_passenger_list.data
    testingIds= test_passenger_list.get_ids()
    myPrediction = session.run(prediction_argmax,
                                feed_dict={xplaceholder: testing_x, lambdaPlaceHolder: 1.0})

    myPrediction = np.asmatrix(myPrediction).T
    data = np.hstack((testingIds.astype(int),myPrediction))

    df = pd.DataFrame(data=data,columns = ['PassengerId','Survived'])
    df.Survived = df.Survived.astype(int)
    df.PassengerId = df.PassengerId.astype(int)
    df.to_csv('output.csv', index=False)

    coord.request_stop()


