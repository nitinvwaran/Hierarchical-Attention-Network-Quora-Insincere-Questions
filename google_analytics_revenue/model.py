import tensorflow as tf
import pandas as pd
import numpy as np
import shutil,os


from sklearn.model_selection import train_test_split



def read_train_make_split(file_name,dev_split):

    data_frame = pd.read_csv(file_name,low_memory=False)
    data_frame.drop('Unnamed: 0',axis=1,inplace=True)
    y = data_frame.loc[:,'is_revenue']
    X_train, X_dev, y_train,y_dev = train_test_split(data_frame,y,test_size=dev_split,stratify=y,random_state=666)

    return X_train, X_dev, y_train,y_dev

def build_optimiser_cost(logits,sigmoids,num_labels):

    pos_wt = 1 # increase to give more weight to the positive class

    with tf.name_scope('cross_entropy'):

        # This is the labels tagged to the dataset
        ground_truth_input = tf.placeholder(tf.float32,shape = [None], name='groundtruth_input')
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')

        # Binary cross-entropy loss
        weighted_cross_entropy_mean = tf.nn.weighted_cross_entropy_with_logits(targets=ground_truth_input, logits=logits,pos_weight=pos_wt,name="loss_binary_xe")
        loss = tf.reduce_mean(weighted_cross_entropy_mean ,name="cross_entropy_loss")

        train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(loss)

        # Construct the confusion matrix

        threshold = tf.constant(0.5,dtype=tf.float32,name="threshold")
        auc = tf.metrics.auc(tf.convert_to_tensor(ground_truth_input, dtype=tf.float32), sigmoids, summation_method='minoring')

        predictions = tf.cast(tf.math.greater(sigmoids,threshold),dtype=tf.int8)
        #correct_prediction = tf.equal(predictions, ground_truth_input, name='correct_prediction')
        confusion_matrix = tf.confusion_matrix(
            ground_truth_input, predictions, num_classes= num_labels, name = "confusion_matrix")
        #evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="eval_step")


        return ground_truth_input, learning_rate_input, train_step, confusion_matrix, weighted_cross_entropy_mean, loss , predictions, auc



def build_graph(fc1_units, fc2_units,fc3_units,dropout_1,dropout_2,dropout_3):

    with (tf.variable_scope("input_layer")):

        inputs = tf.placeholder(shape=[None,257],name="input_train",dtype=tf.float32)

    with (tf.variable_scope("fc_layers")):

        xavier_init_fc_1 = tf.contrib.layers.xavier_initializer()
        xavier_init_fc_2 = tf.contrib.layers.xavier_initializer()
        xavier_init_fc_3 = tf.contrib.layers.xavier_initializer()


        fc1 = tf.layers.dense(inputs,fc1_units,activation=tf.nn.relu,name="layer_fc1",kernel_initializer=xavier_init_fc_1)
        fc1_drop = tf.nn.dropout(fc1,dropout_1)

        #fc2 = tf.layers.dense(fc1_drop, fc2_units, activation=tf.nn.relu, name="layer_fc2",kernel_initializer=xavier_init_fc_2)
        #fc2_drop = tf.nn.dropout(fc2, dropout_2)

        #fc3 = tf.layers.dense(fc2_drop, fc3_units, activation=tf.nn.relu, name="layer_fc3",kernel_initializer=xavier_init_fc_3)
        #fc3_drop = tf.nn.dropout(fc3, dropout_3)

    with (tf.variable_scope("sigmoid_layer")):

        xavier_init_fc_4 = tf.contrib.layers.xavier_initializer()
        wt = tf.get_variable(name="lreg_wt",shape=[fc3_units,1],initializer=xavier_init_fc_4)
        bias = tf.get_variable(name="lreg_bias",shape=[1],initializer=tf.zeros_initializer())

        logits = tf.add(tf.matmul(fc1_drop,wt),bias)
        logits_sq = tf.squeeze(logits,axis=1,name="logits") # to match the ground truth labels shape

        sig = tf.nn.sigmoid(logits_sq)


    return inputs, logits_sq, sig



def get_mini_batch(X_train_0,X_train_1,mini_batch_size):

    # Mini batch size should be even
    if mini_batch_size % 2 != 0:
        mini_batch_size += 1

    sample_0 = X_train_0.sample(n = int(mini_batch_size / 2))
    sample_1 = X_train_1.sample(n= int(mini_batch_size / 2))

    mini_batch_sample = pd.concat([sample_0,sample_1],ignore_index=True)
    mini_batch = mini_batch_sample.sample(frac=1).reset_index(drop=True) # reshuffles

    return mini_batch


def train_model(X_train,X_dev,y_train,y_dev):

    fc1_units = 512
    fc2_units = 512
    fc3_units = 512

    dropout_1 = 1.0
    dropout_2 = 1.0
    dropout_3 = 1.0

    num_epochs = 2000
    num_labels = 2

    learning_rate = 0.001

    mini_batch_size = 128

    X_dev.drop('is_revenue', axis=1, inplace=True)

    X_dev_npy = X_dev.as_matrix()
    nans_index = np.isnan(X_dev_npy)
    X_dev_npy[nans_index] = 0

    y_dev_npy = y_dev.as_matrix()

    print ('Sum is:' + str(y_dev_npy.sum()))

    train_tensorboard_dir = '/home/nitin/Desktop/google_analytics/google_analytics_revenue/train_tensorboard/'
    valid_tensorboard_dir = '/home/nitin/Desktop/google_analytics/google_analytics_revenue/valid_tensorboard/'


    with tf.Graph().as_default() as gr:

        inputs, logits, sig = build_graph(fc1_units,fc2_units,fc3_units,dropout_1,dropout_2,dropout_3)
        ground_truth_input, learning_rate_input, train_step, confusion_matrix, weighted_cross_entropy_mean, loss, predictions , auc= \
        build_optimiser_cost(logits,sig,num_labels)

    with tf.Session(graph=gr) as sess:

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        X_train_0 = X_train.loc[X_train['is_revenue'] == 0]
        X_train_1 = X_train.loc[X_train['is_revenue'] != 0]

        # Tensorboard init
        if (os.path.exists(train_tensorboard_dir)):
            shutil.rmtree(train_tensorboard_dir)
        os.mkdir(train_tensorboard_dir)
        if (os.path.exists(valid_tensorboard_dir)):
            shutil.rmtree(valid_tensorboard_dir)
        os.mkdir(valid_tensorboard_dir)

        train_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_tensorboard_dir)

        xent_counter = 0

        for i in range(1,num_epochs + 1):

            # Get mini batch for the epoch
            mini_batch = get_mini_batch(X_train_0=X_train_0, X_train_1=X_train_1, mini_batch_size=mini_batch_size)
            mini_batch_y = mini_batch.loc[:, 'is_revenue']
            mini_batch.drop('is_revenue', axis=1, inplace=True)

            mini_batch_npy = mini_batch.as_matrix()
            mini_batch_npy_y = mini_batch_y.as_matrix()

            nans_index = np.isnan(mini_batch_npy)
            mini_batch_npy[nans_index] = 0

            print ('Mini-batch retrieved.')

            # Put inputs through the graph
            _, l, conf_matrix, wt_mean = sess.run(
                [
                    train_step, loss, confusion_matrix, weighted_cross_entropy_mean

                ],
                feed_dict={
                    inputs: mini_batch_npy,
                    ground_truth_input: mini_batch_npy_y,
                    learning_rate_input: learning_rate,
                })

            # Write loss to tensorflow for each batch
            xent_train_summary = tf.Summary(
                value=[tf.Summary.Value(tag="cross_entropy_avg", simple_value=l)])
            xent_counter += 1
            train_writer.add_summary(xent_train_summary, xent_counter)

            print ('The loss after batch ' + str(i) + ' is:' + str(l))

            if (i % 10 == 0): # dev metrics after 10 epochs

                conf_matrix, _auc = sess.run(
                    [
                        confusion_matrix, auc
                    ],
                    feed_dict={
                        inputs: X_dev_npy,
                        ground_truth_input: y_dev_npy,
                    })

                #print('Validation Confusion Matrix: ' + '\n' + str(conf_matrix))
                #true_pos = np.sum(np.diag(conf_matrix))
                #all_pos = np.sum(conf_matrix)
                #print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))
                #print('Validation data points:' + str(all_pos))

                # Write validation accuracy to validation tensorboard
                #acc_valid_summary = tf.Summary(
                #    value=[tf.Summary.Value(tag="acc_valid_summary", simple_value=float(true_pos / all_pos))])
                #valid_writer.add_summary(acc_valid_summary, i / 10)


                print('Validation AUC on batch ' + str(i /10) + ' is:' + str(_auc))
                auc_valid_summary = tf.Summary(value=[tf.Summary.Value(tag="auc_valid_summary", simple_value=_auc)])
                valid_writer.add_summary(auc_valid_summary, i / 10)



def main():

    file_name = '/home/nitin/Desktop/google_analytics/google_analytics_revenue/train_mod1_v2.csv'
    dev_split = 0.10
    X_train,X_dev,y_train,y_dev = read_train_make_split(file_name,dev_split)


    train_model(X_train,X_dev,y_train,y_dev)



main()