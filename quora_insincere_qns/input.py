import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import pandas as pd
import re
import numpy as np
import tensorflow as tf


cutoff_shape = 199999
glove_dim = 300



def load_glove_vectors(file):

    wts = []
    glove_dict = {}


    i = 0

    with open (file,'r') as f:

        for index, line in enumerate(f):
            l = line.split(' ')

            #add to the global dictionary
            glove_dict[str(l[0]).strip().lower()] = index
            del l[0]
            wts.append(l)
            i += 1
            if (i > cutoff_shape): # for dev purposes only
                break


    # contains the word embeddings. assumes indexes start from 0-based in the txt file
    weights = np.asarray(wts)

    assert weights.shape[1] == glove_dim
    assert weights.shape[0] == cutoff_shape + 1

    return glove_dict, weights


def read_train_test_words(train_file,test_file, glove_file):

    #UNK = 2196017
    UNK = 200000

    qn_idxs = []

    glove_dict, _ = load_glove_vectors(glove_file)
    train_df = pd.read_csv(train_file, low_memory=False)
    #test_df = pd.read_csv(test_file, low_memory=False)
    #train_qns = train_df.iloc[:,1]

    print ('read train file')
    for index, item in train_df.iterrows():

        qn = item[1].split(' ')
        qn_ls = [re.sub('[^A-Za-z0-9]+', '', q) for q in qn]
        qn_ls = [x.lower() for x in qn_ls]# if x not in stop_words]
        print (qn_ls)

        qn_idx = [glove_dict[x] if (x in glove_dict.keys()) else UNK for x in qn_ls]
        print (qn_idx)
        qn_idxs.append(qn_idx)


    qn_npy = np.asarray(qn_idxs)



def build_graph():

    embedding_const = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[cutoff_shape + 1, glove_dim]),trainable=False,name="embedding_const")

    embedding_placeholder = tf.placeholder(dtype=tf.float32,shape=[cutoff_shape + 1,glove_dim])
    embedding_init = embedding_const.assign(embedding_placeholder)




    return embedding_init, embedding_placeholder


def build_session(glove_embed_file):

    # Build the word embeddings
    _, weights = load_glove_vectors(glove_embed_file)

    with tf.Graph().as_default() as gr:
        embed_init, embed_placeholder = build_graph()


    with tf.Session(graph=gr) as sess:

        sess.run(tf.global_variables_initializer())
        embeds = sess.run(embed_init,feed_dict = {embed_placeholder: weights})

        assert embeds.shape[0] == cutoff_shape + 1
        assert embeds.shape[1] == glove_dim


def main():
    glove_vectors_file = '/home/nitin/Desktop/kaggle_data/all/embeddings/glove.840B.300d/glove.840B.300d.txt'
    train_data = '/home/nitin/Desktop/kaggle_data/all/train.csv'
    test_data = '/home/nitin/Desktop/kaggle_data/all/test.csv'

    #load_glove_vectors(glove_vectors_file)
    #read_train_test_words(train_data,test_data,glove_vectors_file)

    build_session(glove_vectors_file)


main()








