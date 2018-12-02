import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))

import pandas as pd
import re
import numpy as np
np.set_printoptions(threshold=np.nan)
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

            #add to the dictionary
            glove_dict[str(l[0]).strip().lower()] = index
            del l[0]
            wts.append(l)
            i += 1
            if (i > cutoff_shape): # for dev purposes only
                break


    # contains the word embeddings. assumes indexes start from 0-based in the txt file
    # Add the UNK
    unk_wt = np.random.randn(glove_dim)
    wts.append(unk_wt.tolist())
    # Add the _NULL
    null_wt = np.zeros(glove_dim)
    wts.append(null_wt.tolist())

    weights = np.asarray(wts)

    assert weights.shape[1] == glove_dim
    assert weights.shape[0] == cutoff_shape + 3

    return glove_dict, weights


def read_questions(train_file,test_file, glove_file):

    #UNK = 2196017
    UNK = cutoff_shape + 1
    _NULL = cutoff_shape + 2

    glove_dict, _ = load_glove_vectors(glove_file)
    train_df = pd.read_csv(train_file, low_memory=False)
    #test_df = pd.read_csv(test_file, low_memory=False)
    #train_qns = train_df.iloc[:,1]

    qn_ls_word_idx = []
    qn_batch_len = []
    max_sentence_len = -1

    for index, item in train_df.iterrows():

        if (index > 1000): # dev purposes only
            break

        qn1 = item[1].split('. ') # Extract the sentences
        #print (qn1)
        qn2 = [x.split('? ') for x in qn1]
        #print (qn2)
        qn3 = [x for y in qn2 for x in y if x != '']
        #print (qn3)
        qn_ls = [re.sub('[^A-Za-z0-9 ]+', '', q) for q in qn3]
        qn_ls = [x.lower() for x in qn_ls]  # if x not in stop_words]
        #print (qn_ls)
        # word level tokens
        qn_ls_word = [x.split(' ') for x in qn_ls]
        #print (qn_ls_word)

        for y in qn_ls_word:
            tmp = [glove_dict[x] if (x in glove_dict.keys()) else UNK for x in y]

            if (len(tmp) > max_sentence_len):
                max_sentence_len = len(tmp)

            qn_ls_word_idx.append(tmp)
            qn_batch_len.append(len(tmp))

    # Now we have max_len, a flattened sentence matrix, and batch sequence length for dynamic rnn.
    # Apply the _null padding.
    for item in qn_ls_word_idx:
        item += [_NULL] * (max_sentence_len - len(item))

    qn_npy = np.asarray(qn_ls_word_idx)

    return qn_npy, qn_batch_len,max_sentence_len


def build_graph(max_sentence_len):

    def sparse_softmax(T):

        # Creating partition based on condition:
        condition_mask = tf.cast(tf.equal(T, 0.), tf.int32)
        partitioned_T = tf.dynamic_partition(T, condition_mask, 2)
        # Applying the operation to the target partition:
        partitioned_T[0] = tf.nn.softmax(partitioned_T[0],axis=0)


        # Stitching back together, flattening T and its indices to make things easier::
        condition_indices = tf.dynamic_partition(tf.range(tf.size(T)), tf.reshape(condition_mask, [-1]), 2)
        res_T = tf.dynamic_stitch(condition_indices, partitioned_T)
        res_T = tf.reshape(res_T, tf.shape(T))

        return res_T


    gru_units = 10
    output_size = 10
    cell_fw = tf.nn.rnn_cell.GRUCell(gru_units)
    cell_bw = tf.nn.rnn_cell.GRUCell(gru_units)

    # First fetch the pre-trained word embeddings into variable
    # + 2 because of the UNK weight and the 0-index start
    with tf.variable_scope("embedding_layer"):
        embedding_const = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[cutoff_shape + 3, glove_dim]),trainable=False,name="embedding_const")
        embedding_placeholder = tf.placeholder(dtype=tf.float32,shape=[cutoff_shape + 3,glove_dim],name="embedding_placeholder")
        embedding_init = embedding_const.assign(embedding_placeholder)

    # Now load the inputs and convert them to word vectors
    with tf.variable_scope("layer_inputs"):
        inputs = tf.placeholder(dtype=tf.int32, shape=[None,max_sentence_len],name="input")
        inputs_embed = tf.nn.embedding_lookup(embedding_init,inputs,name="input_embed")
        batch_sequence_lengths = tf.placeholder(dtype=tf.int32,name="sequence_length")


    with tf.variable_scope("layer_word_hidden_states"):
        ((fw_outputs,bw_outputs),
         _) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embed,
                                            sequence_length=batch_sequence_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            ))
    outputs_hidden = tf.concat((fw_outputs, bw_outputs), 2)

    with tf.variable_scope("layer_word_attention"):

        initializer = tf.contrib.layers.xavier_initializer()

        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)

        input_projection = tf.contrib.layers.fully_connected(outputs_hidden, output_size,
                                                  activation_fn=tf.nn.tanh)
        vector_attn = tf.tensordot(input_projection,attention_context_vector,axes=[[2],[0]],name="vector_attn")
        attn_softmax = tf.map_fn(lambda batch:
                               sparse_softmax(batch)
                               , vector_attn, dtype=tf.float32)

        attn_softmax = tf.expand_dims(input=attn_softmax,axis=2,name='attn_softmax')

        weighted_projection = tf.multiply(outputs_hidden, attn_softmax)
        outputs = tf.reduce_sum(weighted_projection, axis=1)

    return embedding_init, embedding_placeholder, \
           inputs, inputs_embed, batch_sequence_lengths,\
           vector_attn, attn_softmax, \
           weighted_projection, outputs, outputs_hidden



def build_session(inputs_npy, glove_embed_file, max_sentence_len, qn_batch_len):

    # Build the word embeddings
    _, weights = load_glove_vectors(glove_embed_file)

    with tf.Graph().as_default() as gr:
        embed_init, embed_placeholder, inputs,\
        input_embed, batch_sequence_lengths ,\
        vector_attn, attn_softmax, \
        weighted_projection, outputs, outputs_hidden  = build_graph(max_sentence_len)


    with tf.Session(graph=gr) as sess:

        sess.run(tf.global_variables_initializer())
        embeds, input_embd, out, attn , weighted, outs, outs_hidden  = \
            sess.run([embed_init,input_embed, vector_attn, attn_softmax,
                      weighted_projection,outputs, outputs_hidden ], feed_dict = {
                embed_placeholder: weights,
                inputs : inputs_npy,
                batch_sequence_lengths : qn_batch_len
        })

        assert embeds.shape[0] == cutoff_shape + 3
        assert embeds.shape[1] == glove_dim

        assert input_embd.shape[2] == glove_dim
        assert input_embd.shape[1] == max_sentence_len


        print(out.shape)
        print (attn.shape)

        print(weighted.shape)
        print(outs_hidden.shape)
        print(outs.shape)


def main():
    glove_vectors_file = '/home/nitin/Desktop/kaggle_data/all/embeddings/glove.840B.300d/glove.840B.300d.txt'
    train_data = '/home/nitin/Desktop/kaggle_data/all/train.csv'
    test_data = '/home/nitin/Desktop/kaggle_data/all/test.csv'

    #load_glove_vectors(glove_vectors_file)
    #read_train_test_words(train_data,test_data,glove_vectors_file)

    #build_session(glove_vectors_file)
    qn_npy, qn_batch_len, max_len = read_questions(train_data,test_data,glove_vectors_file)
    build_session(qn_npy,glove_vectors_file,max_len,qn_batch_len)





main()








