import pandas as pd
#pd.set_option('display.max_columns',10)

import re, os, shutil
import numpy as np
#np.set_printoptions(threshold=np.nan)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import shuffle

import tensorflow as tf


# Some global variables
# used constants
cutoff_shape = 2196017          # number of tokens in glove word embeddings file
glove_dim = 300                 # number of glove dimensions
max_seq_len = 122               # determined as the maximum # of words in a sentence. This is unused parameter.

# hyperparameter
cutoff_seq = 40                 # Cutoff the number of words in a sentence. Any sentences with fewer words are padded.

#unused constant
max_sent_seq_len = 12           # determined as maximum of 12 sentences in a doc across train and test data. Unused parameter

# hyperparameter
sent_cutoff_seq = 3             # cutoff the number of sentences in a document. Any documents with fewer sentences are padded.

# hyperparameter
pos_wt = 0.25                   # weights for the loss function.



def load_glove_vectors(memmap_loc, glove_file, reload_mmap = False):

    """
    Loads the glove vectors from a memory mapped embedding file, and returns a dictionary of the indexes of the glove embeddings

    :param memmap_loc: Location of the memory mapped file in disk that stores the glove embeddings on disk
    :param glove_file: location of the glove file on your machine
    :param reload_mmap: Whether to reload the memory map ni the disk.
    :return: A dictionary where keys are the token words and values are the indexes.
    """

    glove_dict = {}
    i = 0
    mmap = None

    if (reload_mmap):
        mmap = np.memmap(memmap_loc, dtype='float32', mode='w+', shape=(cutoff_shape + 2, glove_dim))

    with open (glove_file,'r') as f:

        for index, line in enumerate(f):
            l = line.split(' ')

            #add to the dictionary
            glove_dict[str(l[0])] = index

            # Add to memmap
            if (reload_mmap):
                del l[0]
                mmap[index,:] = l

            i += 1

    # contains the word embeddings. assumes indexes start from 0-based in the txt file
    if (reload_mmap):

        # Add the UNK - in case token cant be found in glove file.
        #unk_wt = np.random.randn(glove_dim)
        unk_wt = np.zeros(glove_dim)
        mmap[i:] = unk_wt
        i += 1

        # Add the _NULL - for padding purposes.
        null_wt = np.zeros(glove_dim)
        mmap[i:] = null_wt
        mmap.flush()

    return glove_dict


def get_train_df_glove_dict(train_file, glove_file,mmap_loc, is_training = True, reload_mmap=False):

    """

    Method to do the train and dev split and retuen both datasets, along with the glove dictionary

    :param train_file: training data file
    :param glove_file: glove embedding data file
    :param mmap_loc: location of memory map on disk
    :param is_training: flag to tell whether the method is called during training or inference process
    :param reload_mmap: flag to tell whether to reload the memory map on disk. E.g Not reloaded during inference.
    :return: Train dataframe, dev dataframe, and glove dictionary if training, or just the glove dictionary if inference.
    """


    # creates the mmap
    glove_dict = load_glove_vectors(mmap_loc,glove_file,reload_mmap)

    if (is_training):

        df = pd.read_csv(train_file, low_memory=False)

        y = df.loc[:, 'target']
        X_train, X_dev, y_train, y_dev = train_test_split(df, y, test_size=0.01, stratify=y, random_state=42)

        X_train_0 = X_train.loc[X_train['target'] == 0]
        X_train_1 = X_train.loc[X_train['target'] == 1]

        X_train_0_sample = X_train_0.sample(n=X_train_0.shape[0])   # Number of training samples with 0 label
        X_train_1_sample = X_train_1.sample(n=X_train_1.shape[0])   # Number of training samples with 1 label

        X_dev_0 = X_dev.loc[X_dev['target'] == 0]
        X_dev_1 = X_dev.loc[X_dev['target'] == 1]

        X_dev_0_sample = X_dev_0.sample(n=X_dev_0.shape[0])
        X_dev_1_sample = X_dev_1.sample(n=X_dev_1.shape[0])

        X_train_f = pd.concat([X_train_0_sample,X_train_1_sample],axis=0)
        X_train_f = shuffle(X_train_f)
        X_dev_f = pd.concat([X_dev_0_sample, X_dev_1_sample], axis=0)

        # Dump the validation data out here
        valid_dump = '/home/nitin/Desktop/kaggle_data/all/valid_dump.csv'
        X_dev.to_csv(valid_dump,index=False)

        return X_train_f, X_dev_f, glove_dict

    else:

        return None, None , glove_dict



def process_questions(qn_df, glove_dict, mmap, is_training = True):

    """

    Process each document, unrolling the sentences in each document into a flat matrix
    before looking up the word embeddings data for each word in each sentence.
    Returns among other things (see below), a 3-D word embedding tensor ready to be fed to the 'input' placeholder.

    Add other pre-processing steps here (e.g spelling corrections..)

    :param qn_df: Dataframe of raw document data - for the competition's sake, question data
    :param glove_dict: dictionary of embeddings with key = token and value = index in embedding file
    :param mmap: location of memory map file on disk
    :param is_training: flag whether training run or inference run
    :return: qn_ls_word_embd, qn_batch_len, sentence_batch_len, sentence_batch_len_2, y_len_2

    1) qn_ls_word_embd: the embedding in 3-D tensor for the qn_df dataframe after unrolling all the sentences in the document
    2) qn_batch_len : number of words in each of the unrolled sentences.
    3) sentence_batch_len : number of sentences in a document
    4) sentence_batch_len_2 : number of sentences in a document, but the value is broadcast to all the sentences in the document. It will become clearer why, later on.
    5) y_len_2 : labels attached to flattened sentences in a document, given the label is at document level. During inference, this will contain the normalized row ID instead of the label. WHy, will become apparent later.
    """


    UNK = cutoff_shape
    _NULL = cutoff_shape + 1
    sentence_batch_len = []
    sentence_batch_len_2 = []
    y_len_2 = []

    qn_ls_word_idx = []
    qn_batch_len = []

    l = 0

    '''
    # Stopwords can be removed from data if desired.
    
    stop_words = ['how', 'were', 'through', 'up', 'ma', 'because', 'his', "hasn't", 'myself', 'that', 'then', 'm', 'haven', 'during',
     'being', 'needn', 'whom', 'did', 'the', 'very', 'd', 'as', 'their', 'ours', 'was', 'be', 'themselves', "mustn't",
     'she', 'or', 'but', 'yourself', 'if', 'had', 'not', 'itself', 'than', 'such', 'having', 'we', 'again', 'about',
     'this', 'with', 'any', 'on', 'has', 'a', 'there', 'so', 'all', 'both', 'him', 'its', 'yourselves', "couldn't",
     'by', 'they', "mightn't", 'mightn', "wouldn't", 're', 'you', 'above', 'more', 'while', "you've", 'after', 'mustn',
     'shan', 's', 'some', 'in', 'will', 'and', 'same', 'hadn', "haven't", 'weren', "don't", 'herself', "won't", 'of',
     'down', 'aren', 'ourselves', 'wasn', 'your', 'just', 'yours', 'out', 'have', 'isn', 'most', 'until', 'can', 'it',
     "didn't", 'don', "you're", 'over', 't', 'couldn', 'who', 'these', 'below', 'each', 'should', 'hers', 'other',
     "she's", "shan't", 'shouldn', 've', "it's", 'y', 'wouldn', 'into', 'what', 'does', 'where', 'to', "you'll",
     "weren't", 'at', "that'll", 'do', 'further', 'been', 'are', 'i', 'which', 'own', 'from', 'why', 'between',
     "shouldn't", 'nor', "hadn't", "you'd", 'is', 'an', 'am', 'o', 'didn', 'for', 'them', 'only', 'hasn', 'me',
     "wasn't", 'he', "aren't", 'too', "should've", 'now', 'ain', 'few', 'll', 'her', 'against', 'theirs', 'off',
     'doing', "doesn't", 'won', "isn't", 'once', 'here', 'my', 'before', 'those', 'under', 'himself', 'doesn', 'when',
     "needn't", 'no', 'our']
    '''


    for index, item in qn_df.iterrows():


        qn1 = item[1].split('. ') # Extract the sentences
        #print (qn1)
        qn2 = [x.split('? ') for x in qn1]
        #print (qn2)
        qn3 = [x for y in qn2 for x in y if x != '']
        #print (qn3)

        qn_ls_0 = [q.replace('/',' ') for q in qn3]
        qn_ls_0 = [re.sub(' +', ' ',q) for q in qn_ls_0]
        qn_ls = [re.sub('[^A-Za-z0-9\' ]+', '', q) for q in qn_ls_0]
        #print (qn_ls)


        # Cutoff
        if (len(qn_ls) > sent_cutoff_seq):
            qn_ls = qn_ls[:sent_cutoff_seq]

        sentence_batch_len.append(len(qn_ls))

        # word level tokens
        qn_ls_word = [x.split(' ') for x in qn_ls]

        # Bit misnomer. qn_ls_word is still array of sentences in each document / answer
        for y in qn_ls_word:

            if (is_training):
                y_len_2.append(item[2]) # 'elongate' the target variable. This will also need to be re-stitched later on.
            else:
                y_len_2.append(item[0]) # this is called during inference. The same trick is played on the key ID during the graph.

            tmp = [glove_dict[x] if (x in glove_dict.keys()) else UNK for x in y]
            qn_ls_word_idx.append(tmp)


            if (len(tmp) > cutoff_seq):
                qn_batch_len.append(cutoff_seq)
            else:
                qn_batch_len.append(len(tmp))

            sentence_batch_len_2.append(len(qn_ls))

    # Now we have max_len, a flattened sentence matrix, and batch sequence length for dynamic rnn.
    # Apply the _null padding.
    batch_shape = len(qn_ls_word_idx)
    qn_ls_word_embd = np.empty((batch_shape, cutoff_seq, glove_dim))

    for item in qn_ls_word_idx:
        item += [_NULL] * (max_seq_len - len(item))

        # Apply a cutoff
        item = item[:cutoff_seq]

        # Word embedding lookup from memmapped file
        tmp_npy = mmap[item]
        tmp_npy = np.expand_dims(tmp_npy,axis=0)

        qn_ls_word_embd[l] = tmp_npy

        l += 1

    return qn_ls_word_embd, qn_batch_len, sentence_batch_len, sentence_batch_len_2, y_len_2 # Flattened qn_lengths, and sentence_len at document level


def build_graph(max_sentence_len):

    """

    :param max_sentence_len: cutoff of number of words in a sentence
    :param keep_1: keep probability for word level n-gram convolutional feature extraction
    :param keep_2: keep probability for sentence level convolutional feature extraction
    :param keep_att_1: keep probability for word level hierarchical features
    :param keep_att_2: keep probability for sentence level hierarchical features
    :param keep_fc:  keep probability for final dense layer
    :return: the relevant graph variables required for the session machinery


    There are two separate networks in the graph:
    1) The first is the hierarchical attention network (HAN), proposed by Yang et al (2016)
    2) The second is borrowed from the ideas from Sentence Classification using CNN proposed by Yoon Kim (2014)

    for more info on the hierarchical attention network initial implementation, see:
    https://www.linkedin.com/pulse/implementation-hierarchical-attention-network-nitin-venkateswaran?articleId=6483388258968539136#comments-6483388258968539136&trk=prof-post

    The additions for the CNN-based sentence classifications were made later
    The CNN layers also operate in hierarchical fashion, first:
    1) extracting n-gram features from each unrolled sentence in a document and taking a global maxpool over time for each sentence/n-gram (3,4 and 5 grams are extracted using convolutional windows)
    2) the n-gram features are then concatenated for each sentence. The sentences are rolled up into their document
    3) a max-pool for all the concatenated features over all the sentences in a document are taken. It may be possible to implement another CNN set of layers here (but this was eventually commented out)

    Features from the HAN and the CNN layers and finally stacked, and sent through a dense layer, before logistic regression is applied.

    """

    keep_fc = tf.placeholder(dtype=tf.float32, name="keep_1")
    keep_conv = tf.placeholder(dtype=tf.float32,name='keep_conv')

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

    # hyperparameters
    gru_units = 50         # Feature maps for the GRU / LSTM units at word and sentence levels
    output_size = 50       # Size of the word attention context vector

    gru_units_sent = 50
    output_size_sent = 50  # Size of the Sentence attention context vector

    cell_fw = tf.nn.rnn_cell.GRUCell(gru_units)
    cell_bw = tf.nn.rnn_cell.GRUCell(gru_units)

    cell_sent_fw = tf.nn.rnn_cell.GRUCell(gru_units_sent)
    cell_sent_bw = tf.nn.rnn_cell.GRUCell(gru_units_sent)

    '''
    Load the embeddings directly from numpy
    '''
    with tf.variable_scope("layer_inputs"):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,max_sentence_len,glove_dim],name="input")
        batch_sequence_lengths = tf.placeholder(dtype=tf.int32,name="sequence_length")


    '''
    Word level n-gram CNN extraction on the lines on Yoon Kim's paper (2014)
    '''

    with tf.variable_scope('layer_conv_1D'):

        # Does feature selection using n-grams + CNN for each sentence in a document.

        # hyperparameters
        feature_map_size = 100
        kernel_1 = 3
        kernel_2 = 4
        kernel_3 = 5

        l1w_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        l2w_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        l3w_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)

        first_conv_layer = tf.layers.conv1d(inputs=inputs, filters=feature_map_size,
                                            kernel_size=kernel_1, strides=1, padding='valid',
                                            activation=tf.nn.relu, use_bias=True, kernel_initializer=l1w_init,
                                            bias_initializer=tf.zeros_initializer(), name="first_conv_layer")

        first_maxpool_layer = tf.reduce_max(first_conv_layer, axis=1, keepdims=True, name="conv_layer_1_max")
        first_maxpool_layer = tf.squeeze(first_maxpool_layer)

        second_conv_layer = tf.layers.conv1d(inputs=inputs, filters=feature_map_size,
                                             kernel_size=kernel_2, strides=1, padding='valid',
                                             activation=tf.nn.relu, use_bias=True, kernel_initializer=l2w_init,
                                             bias_initializer=tf.zeros_initializer(), name="second_conv_layer")


        second_maxpool_layer = tf.reduce_max(second_conv_layer, axis=1, keepdims=True, name="conv_layer_2_max")
        second_maxpool_layer = tf.squeeze(second_maxpool_layer)

        third_conv_layer = tf.layers.conv1d(inputs=inputs, filters=feature_map_size,
                                            kernel_size=kernel_3, strides=1, padding='valid',
                                            activation=tf.nn.relu, use_bias=True, kernel_initializer=l3w_init,
                                            bias_initializer=tf.zeros_initializer(), name="third_conv_layer")

        third_maxpool_layer = tf.reduce_max(third_conv_layer, axis=1, keepdims=True, name="conv_layer_3_max")
        third_maxpool_layer = tf.squeeze(third_maxpool_layer)

        output_features_conv = tf.concat(values=[first_maxpool_layer, second_maxpool_layer, third_maxpool_layer], axis = 1,
                                         name="concat_conv")

        output_features_conv.set_shape(shape=[None,feature_map_size * 3])

        output_features_conv = tf.nn.dropout(output_features_conv,keep_prob=keep_conv)


    '''
    This is the word level annotation encoder for each sentence in a document
    '''
    with tf.variable_scope("layer_word_hidden_states"):
        ((fw_outputs,bw_outputs),
         _) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs,
                                            sequence_length=batch_sequence_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            ))
    outputs_hidden = tf.concat((fw_outputs, bw_outputs), 2)

    '''
    This uses the word level attention context vector to extract 
    '''
    with tf.variable_scope("layer_word_attention"):

        initializer = tf.contrib.layers.xavier_initializer()

        # Big brain #1
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size ],
                                                   initializer=initializer,
                                                   dtype=tf.float32)

        input_projection = tf.contrib.layers.fully_connected(outputs_hidden, output_size ,
                                                  activation_fn=tf.nn.tanh)
        vector_attn = tf.tensordot(input_projection,attention_context_vector,axes=[[2],[0]],name="vector_attn")
        attn_softmax = tf.map_fn(lambda batch:
                               sparse_softmax(batch)
                               , vector_attn, dtype=tf.float32)

        attn_softmax = tf.expand_dims(input=attn_softmax,axis=2,name='attn_softmax')

        weighted_projection = tf.multiply(outputs_hidden, attn_softmax)
        outputs = tf.reduce_sum(weighted_projection, axis=1)


    with tf.variable_scope('layer_gather_conv'):

        tf_padded_final_conv = tf.zeros(shape=[1,sent_cutoff_seq,feature_map_size * 3]) # as 3 windows

        sentence_batch_length_2 = tf.placeholder(shape=[None],dtype=tf.int32,name="sentence_batch_len_2")

        i_conv = tf.constant(1)

        #This rolls up sentences dyamically, each sentence-batch-shape at a time.
        def while_cond_conv (i_conv, tf_padded_final_conv):

            mb_conv = tf.constant(sent_cutoff_seq)
            return tf.less_equal(i_conv,mb_conv)

        def body_conv(i_conv,tf_padded_final_conv):

            tf_mask_conv = tf.equal(sentence_batch_length_2,i_conv)
            tf_slice_conv = tf.boolean_mask(output_features_conv,tf_mask_conv,axis=0)

            tf_slice_reshape_conv = tf.reshape(tf_slice_conv,shape=[-1,i_conv,tf_slice_conv.get_shape().as_list()[1]])
            pad_len_conv = sent_cutoff_seq - i_conv

            tf_slice_padding_conv = [[0,0], [0, pad_len_conv], [0, 0]]
            tf_slice_padded_conv = tf.pad(tf_slice_reshape_conv, tf_slice_padding_conv, 'CONSTANT')

            tf_padded_final_conv = tf.concat([tf_padded_final_conv,tf_slice_padded_conv],axis=0)

            i_conv = tf.add(i_conv,1)

            return i_conv, tf_padded_final_conv

        _, tf_padded_final_2_conv = \
            tf.while_loop(while_cond_conv, body_conv, [i_conv, tf_padded_final_conv],shape_invariants=[i_conv.get_shape(),tf.TensorShape([None,sent_cutoff_seq,feature_map_size * 3])])

    # Give it a haircut
    tf_padded_final_2_conv = tf_padded_final_2_conv[1:,:]


    with tf.variable_scope('layer_gather'):

        tf_padded_final = tf.zeros(shape=[1,sent_cutoff_seq,output_size * 2])
        tf_y_final = tf.zeros(shape=[1,1],dtype=tf.int32)
        sentence_batch_len = tf.placeholder(shape=[None],dtype=tf.int32,name="sentence_batch_len")
        sentence_index_offsets = tf.placeholder(shape=[None,2],dtype=tf.int32,name="sentence_index_offsets")
        #sentence_batch_length_2 = tf.placeholder(shape=[None],dtype=tf.int32,name="sentence_batch_len_2")
        ylen_2 = tf.placeholder(shape=[None],dtype=tf.int32,name="ylen_2")

        i = tf.constant(1)
        
        def while_cond (i, tf_padded_final, tf_y_final):

            mb = tf.constant(sent_cutoff_seq)
            return tf.less_equal(i,mb)

        def body(i,tf_padded_final,tf_y_final):

            tf_mask = tf.equal(sentence_batch_length_2,i)
            tf_slice = tf.boolean_mask(outputs,tf_mask,axis=0)
            tf_y_slice = tf.boolean_mask(ylen_2,tf_mask,axis=0) # reshaping the y to fit the data

            tf_slice_reshape = tf.reshape(tf_slice,shape=[-1,i,tf_slice.get_shape().as_list()[1]])
            tf_y_slice_reshape = tf.reshape(tf_y_slice,shape=[-1,i])
            tf_y_slice_max = tf.reduce_max(tf_y_slice_reshape,axis=1,keep_dims=True) # the elements should be the same across the col
            pad_len = sent_cutoff_seq - i
            tf_slice_padding = [[0,0], [0, pad_len], [0, 0]]
            tf_slice_padded = tf.pad(tf_slice_reshape, tf_slice_padding, 'CONSTANT')
            tf_padded_final = tf.concat([tf_padded_final,tf_slice_padded],axis=0)
            tf_y_final = tf.concat([tf_y_final,tf_y_slice_max],axis=0)
            i = tf.add(i,1)

            return i, tf_padded_final, tf_y_final

        _, tf_padded_final_2, tf_y_final_2 = tf.while_loop(while_cond, body, [i, tf_padded_final, tf_y_final],shape_invariants=[i.get_shape(),tf.TensorShape([None,sent_cutoff_seq,output_size_sent * 2]),tf.TensorShape([None,1])])

    # Give it a haircut
    tf_padded_final_2 = tf_padded_final_2[1:,:]
    tf_y_final_2 = tf_y_final_2[1:,:]


    with tf.variable_scope('layer_sentence_conv'):

        '''
        feature_map_size_sent = 100 # preserve the feature map
        kernel_1_sent = 2
        kernel_2_sent = 3
        #kernel_3_sent = 4

        l1w_init_sent = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        l2w_init_sent = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        l3w_init_sent = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)

        first_conv_layer_sent = tf.layers.conv1d(inputs=tf_padded_final_2_conv, filters=feature_map_size_sent,
                                            kernel_size=kernel_1_sent, strides=1, padding='valid',
                                            activation=tf.nn.relu, use_bias=True, kernel_initializer=l1w_init_sent,
                                            bias_initializer=tf.zeros_initializer(), name="first_conv_layer_sent")

        first_maxpool_layer_sent = tf.reduce_max(first_conv_layer_sent, axis=1, keepdims=True, name="conv_layer_1_max_sent")
        first_maxpool_layer_sent = tf.squeeze(first_maxpool_layer_sent)

        second_conv_layer_sent = tf.layers.conv1d(inputs=tf_padded_final_2_conv, filters=feature_map_size_sent,
                                             kernel_size=kernel_2_sent, strides=1, padding='valid',
                                             activation=tf.nn.relu, use_bias=True, kernel_initializer=l2w_init_sent,
                                             bias_initializer=tf.zeros_initializer(), name="second_conv_layer_sent")

        second_maxpool_layer_sent = tf.reduce_max(second_conv_layer_sent, axis=1, keepdims=True, name="conv_layer_2_max_sent")
        second_maxpool_layer_sent = tf.squeeze(second_maxpool_layer_sent)

        third_conv_layer_sent = tf.layers.conv1d(inputs=tf_padded_final_2_conv, filters=feature_map_size_sent,
                                                  kernel_size=kernel_3_sent, strides=1, padding='valid',
                                                  activation=tf.nn.relu, use_bias=True,
                                                  kernel_initializer=l3w_init_sent,
                                                  bias_initializer=tf.zeros_initializer(),
                                                  name="third_conv_layer_sent")

        third_maxpool_layer_sent = tf.reduce_max(third_conv_layer_sent, axis=1, keepdims=True,
                                                  name="conv_layer_3_max_sent")
        third_maxpool_layer_sent = tf.squeeze(third_maxpool_layer_sent)

        #output_conv_sent = tf.concat([first_maxpool_layer_sent,second_maxpool_layer_sent, third_maxpool_layer_sent],axis=1,name="output_conv_sent")
        output_conv_sent_1 = tf.concat([first_maxpool_layer_sent,second_maxpool_layer_sent],axis=1,name="output_conv_sent")
        output_conv_sent = tf.nn.dropout(output_conv_sent_1,keep_prob=keep_2)
        '''

        output_conv_sent = tf.reduce_max(tf_padded_final_2_conv,axis=1,keepdims=True) # suppose reduce by sum?
        output_conv_sent = tf.squeeze(output_conv_sent)




    with tf.variable_scope('layer_sentence_hidden_states'):

        ((fw_outputs_sent, bw_outputs_sent),
         _) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_sent_fw,
                                            cell_bw=cell_sent_bw,
                                            inputs=tf_padded_final_2,
                                            sequence_length=sentence_batch_len,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            ))
        outputs_hidden_sent = tf.concat((fw_outputs_sent, bw_outputs_sent), 2)

    with tf.variable_scope('layer_sentence_attention'):

        initializer_sent = tf.contrib.layers.xavier_initializer()

        # Big brain #2 (or is this Pinky..?)
        attention_context_vector_sent = tf.get_variable(name='attention_context_vector_sent',
                                                   shape=[output_size_sent ],
                                                   initializer=initializer_sent,
                                                   dtype=tf.float32)

        input_projection_sent = tf.contrib.layers.fully_connected(outputs_hidden_sent, output_size_sent,
                                                             activation_fn=tf.nn.tanh)
        vector_attn_sent = tf.tensordot(input_projection_sent, attention_context_vector_sent, axes=[[2], [0]], name="vector_attn_sent")
        attn_softmax_sent = tf.map_fn(lambda batch:
                                 sparse_softmax(batch)
                                 , vector_attn_sent, dtype=tf.float32)

        attn_softmax_sent = tf.expand_dims(input=attn_softmax_sent, axis=2, name='attn_softmax_sent')

        weighted_projection_sent = tf.multiply(outputs_hidden_sent, attn_softmax_sent)
        outputs_sent = tf.reduce_sum(weighted_projection_sent, axis=1)


     # Add a fully connected layer for impact
    with tf.variable_scope('fc_layers'):

        # more features !
        final_output = tf.concat([output_conv_sent, outputs_sent], axis=1, name="final_concat")
        final_output.set_shape(shape = [None,feature_map_size * 3 + output_size_sent * 2])

        fc1_size = 256 # hyperparameter
        fc1 = tf.contrib.layers.fully_connected(final_output, fc1_size,
                                                             activation_fn=tf.nn.relu)
        fc1_drop = tf.nn.dropout(fc1,keep_fc)

        # more features !

        fc2_size = 256  # hyperparameter
        fc2 = tf.contrib.layers.fully_connected(fc1_drop, fc2_size,
                                                activation_fn=tf.nn.relu)
        fc2_drop = tf.nn.dropout(fc2, keep_fc)


    with tf.variable_scope('layer_classification'):

        wt_init = tf.contrib.layers.xavier_initializer()
        #wt = tf.get_variable(name="wt",shape=[output_size * 2 +  feature_map_size * 3 ,1],initializer=wt_init)
        wt = tf.get_variable(name="wt",shape=[fc2_size ,1],initializer=wt_init)
        #wt = tf.get_variable(name="wt",shape=[output_size * 2,1],initializer=wt_init)
        bias = tf.get_variable(name="bias",shape=[1],initializer=tf.zeros_initializer())


        #logits = tf.layers.dense(final_output,units = 1 , activation=tf.nn.sigmoid,use_bias=True,kernel_initializer = wt_init)
        #logits = tf.add(tf.matmul(outputs_sent,wt),bias)

        logits = tf.add(tf.matmul(fc2_drop,wt),bias)
        logits = tf.squeeze(logits)
        #logits = tf.squeeze(outputs_sent_2)
        probs = tf.sigmoid(logits)
        tf_y_final_2 = tf.squeeze(tf_y_final_2)
        tf_y_final_2 = tf.to_float(tf_y_final_2)


    with tf.variable_scope('cross_entropy'):

        global_step = tf.Variable(0,trainable=False,dtype=tf.int32,name='global_step')

        cross_entropy_mean = tf.nn.weighted_cross_entropy_with_logits(
            targets=tf_y_final_2, logits=logits, pos_weight=pos_wt)
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')

        loss = tf.reduce_mean(cross_entropy_mean, name="cross_entropy_loss")

        train_step = tf.train.AdamOptimizer(
            learning_rate_input).minimize(loss,global_step=global_step)

        #train_step = tf.train.GradientDescentOptimizer(
        #    learning_rate_input).minimize(loss)

        #train_step = tf.train.AdagradOptimizer(
        #    learning_rate_input).minimize(loss)

        #train_step = tf.train.MomentumOptimizer(
        #    learning_rate_input,momentum=0.9).minimize(loss,global_step=global_step)

        predicted_indices = tf.to_int32(tf.greater_equal(logits,0.7)) # hyperparameter
        confusion_matrix = tf.confusion_matrix(
            tf_y_final_2, predicted_indices, num_classes=2, name="confusion_matrix")


    return probs, logits, inputs,batch_sequence_lengths, sentence_batch_len, \
            sentence_index_offsets,  sentence_batch_length_2, tf_y_final_2, ylen_2, \
            learning_rate_input, train_step, confusion_matrix, cross_entropy_mean, \
            loss, global_step, predicted_indices, keep_fc, keep_conv



def build_loss_optimizer(logits):
    # Create the back propagation and training evaluation machinery in the graph.
    pass



def build_session(train_file, glove_file,mmap_loc,chkpoint_dir,train_tensorboard_dir,valid_tensorboard_dir):

    """

    :param train_file: train data file
    :param glove_file: file with the glove embeddings
    :param mmap_loc: memory map location on disk to be written to during training
    :param chkpoint_dir: checkpoint directory
    :param train_tensorboard_dir: directory where training events are written to tensorboard file
    :param valid_tensorboard_dir: directory where events on validation data are written to tensorboard file
    :return: Nothing
    """

    # hyperparameters
    num_epochs = 200000
    mini_batch_size = 64
    learning_rate = 0.001



    if (os.path.exists(train_tensorboard_dir)):
        shutil.rmtree(train_tensorboard_dir)
    os.mkdir(train_tensorboard_dir)

    if (os.path.exists(valid_tensorboard_dir)):
        shutil.rmtree(valid_tensorboard_dir)
    os.mkdir(valid_tensorboard_dir)



    # Build the graph and the optimizer and loss
    with tf.Graph().as_default() as gr:
        final_probs, logits,  inputs, batch_sequence_lengths, sentence_batch_len,\
        sentence_index_offsets, sentence_batch_length_2, tf_y_final_2, ylen_2, \
        learning_rate_input, train_step, confusion_matrix, cross_entropy_mean, \
        loss, global_step, predicted_indices, keep_fc, keep_conv = \
            build_graph(cutoff_seq)

    X_train, X_dev, glove_dict = get_train_df_glove_dict(train_file, glove_file,mmap_loc,reload_mmap = False)

    valid_set_shape = X_dev.shape[0]
    # Open the read mmap
    mmap = np.memmap(mmap_loc, dtype='float32', mode='r', shape=(cutoff_shape + 2, glove_dim))

    with tf.Session(graph=gr,config=tf.ConfigProto(log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restore model and continue
        ckpt = tf.train.get_checkpoint_state(chkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)


        train_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_tensorboard_dir)

        xent_counter = 0

        for i in range(0,num_epochs):

            #if (i == 2000):
            #    learning_rate /= 10 # Step by step learning rate

            X_train_0_sample = X_train.loc[X_train['target'] == 0].sample(n=mini_batch_size)
            X_train_1_sample = X_train.loc[X_train['target'] == 1].sample(n=round(mini_batch_size * 1.0)) # hyperparameter Downsample or upsample

            train_sample = pd.concat([X_train_0_sample,X_train_1_sample],axis=0)
            train_sample = shuffle(train_sample)

            qn_npy, qn_batch_len,  sentence_len, sentence_batch_train_2, y_len_2 = process_questions(train_sample,glove_dict,mmap)
            ylen2_npy = np.asarray(y_len_2)

            # Create a matrix with each row as sentence offsets
            # That gets used to rollup the flattened sentences into their documents
            sentence_offsets = np.cumsum(sentence_len)
            sentence_offsets_2 = np.insert(sentence_offsets,0,0,axis=0)
            sentence_offsets_3 = np.delete(sentence_offsets_2,sentence_offsets_2.shape[0] - 1)
            np_offsets_len = np.column_stack([sentence_offsets_3,sentence_offsets])

            prob, _, train_confusion_matrix, train_loss, y_2, pred_indices = \
            sess.run([final_probs,train_step,confusion_matrix, loss, tf_y_final_2 ,predicted_indices], feed_dict = {
                    inputs : qn_npy,
                    batch_sequence_lengths : qn_batch_len,
                    sentence_batch_len : sentence_len,
                    sentence_index_offsets : np_offsets_len,
                    learning_rate_input : learning_rate,
                    sentence_batch_length_2 : sentence_batch_train_2,
                    ylen_2 : ylen2_npy,
                    keep_fc : 0.5, # Keep probability,
                    keep_conv : 0.5
            })

            print ('Epoch is:' + str(i))
            print ('Training Confusion Matrix')
            print (train_confusion_matrix)
            print ('Train loss')
            print (train_loss)

            true_pos = np.sum(np.diag(train_confusion_matrix))
            all_pos = np.sum(train_confusion_matrix)
            print('Training Accuracy is: ' + str(float(true_pos / all_pos)))
            print('Total data points:' + str(all_pos))

            train_auc = roc_auc_score(y_2,prob,average="weighted")
            print ('Train AUC')
            print (train_auc)

            train_f1 = f1_score(y_2,pred_indices)
            print('Train F1')
            print(train_f1)

            xent_counter += 1

            loss_train_summary = tf.Summary(
                value=[tf.Summary.Value(tag="loss_train_summary", simple_value=train_loss)])
            train_writer.add_summary(loss_train_summary, xent_counter)

            acc_train_summary = tf.Summary(
                value=[tf.Summary.Value(tag="acc_train_summary", simple_value=float(true_pos / all_pos))])
            train_writer.add_summary(acc_train_summary, xent_counter)

            auc_train_summary = tf.Summary(
                value=[tf.Summary.Value(tag="auc_train_summary", simple_value=train_auc)])
            train_writer.add_summary(auc_train_summary, xent_counter)

            f1_train_summary = tf.Summary(
                value=[tf.Summary.Value(tag="f1_train_summary", simple_value=train_f1)])
            train_writer.add_summary(f1_train_summary, xent_counter)

            if (i % 10 == 0):

                print('Saving checkpoint for epoch:' + str(i))
                saver.save(sess=sess, save_path=chkpoint_dir + 'quora_insincere_qns.ckpt',
                           global_step=global_step)


            # Validation machinery
            if (i % 50 == 0):

                valid_conf_matrix = None
                validation_loss = None

                print ('Valid set shape')
                print (valid_set_shape)

                qn_npy_valid, qn_batch_len_valid, sentence_len_valid, sentence_batch_valid_2, y_len_valid_2 = process_questions(X_dev, glove_dict,mmap)

                ylen2_valid_npy = np.asarray(y_len_valid_2)

                sentence_offsets = np.cumsum(sentence_len_valid)
                sentence_offsets_2 = np.insert(sentence_offsets, 0, 0, axis=0)
                sentence_offsets_3 = np.delete(sentence_offsets_2, sentence_offsets_2.shape[0] - 1)
                np_offsets_len = np.column_stack([sentence_offsets_3, sentence_offsets])


                valid_prob, conf_matrix, valid_loss, y_2_valid,val_pred_indices = \
                    sess.run([final_probs,confusion_matrix, loss,tf_y_final_2, predicted_indices], feed_dict={
                        inputs: qn_npy_valid,
                        batch_sequence_lengths: qn_batch_len_valid,
                        sentence_batch_len: sentence_len_valid,
                        sentence_index_offsets: np_offsets_len,
                        sentence_batch_length_2 : sentence_batch_valid_2,
                        ylen_2 : ylen2_valid_npy,
                        keep_fc : 1.0,
                        keep_conv : 1.0
                    })

                if valid_conf_matrix is None:
                    valid_conf_matrix = conf_matrix
                    validation_loss = valid_loss
                else:
                    valid_conf_matrix += conf_matrix
                    validation_loss += valid_loss

                print ('Validation Conf matrix')
                print(valid_conf_matrix)
                print ('Validation Loss')
                print (validation_loss)

                true_pos = np.sum(np.diag(valid_conf_matrix))
                all_pos = np.sum(valid_conf_matrix)
                print('Valid Accuracy is: ' + str(float(true_pos / all_pos)))
                print('Total data points:' + str(all_pos))

                valid_auc = roc_auc_score(y_2_valid, valid_prob,average="weighted")
                print('Valid AUC')
                print(valid_auc)

                valid_f1 = f1_score(y_2_valid,val_pred_indices)
                print('Valid F1')
                print(valid_f1)

                loss_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss_valid_summary", simple_value=validation_loss)])
                valid_writer.add_summary(loss_valid_summary, i / 20)

                acc_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_valid_summary", simple_value=float(true_pos / all_pos))])
                valid_writer.add_summary(acc_valid_summary, i / 20)

                auc_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="auc_valid_summary", simple_value=valid_auc)])
                valid_writer.add_summary(auc_valid_summary, i / 20)

                f1_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="f1_valid_summary", simple_value=valid_f1)])
                valid_writer.add_summary(f1_valid_summary, i / 20)


def inference(test_file, glove_file, mmap_loc, chkpoint_dir, out_file):

    """

    Does inference on a test data set

    :param test_file: test data file
    :param glove_file: file with glove embeddings
    :param mmap_loc: memory map location on disk
    :param chkpoint_dir: where the checkpoint is stored (directory)
    :param out_file: a file with the predictions to be stored on
    :return: Nothing
    """

    reload_mmap = False # Inference

    test_df = pd.read_csv(test_file,low_memory=False)
    batch_final = np.zeros((1,2))

    test_df['qid_num'] = range(0,len(test_df))

    meta_data = test_df[['qid_num','question_text']]
    print (meta_data.shape)
    print (meta_data.head(10))

    threshold = 0.5

    # Build the graph and the optimizer and loss
    with tf.Graph().as_default() as gr:
        final_probs, logits,  inputs, batch_sequence_lengths, sentence_batch_len,\
        sentence_index_offsets, sentence_batch_length_2, tf_y_final_2, ylen_2, \
        learning_rate_input, train_step, confusion_matrix, cross_entropy_mean, \
        loss, global_step, predicted_indices, keep_fc, keep_conv = \
            build_graph(cutoff_seq)

    _, _, glove_dict = get_train_df_glove_dict(test_file, glove_file, mmap_loc,is_training=False,reload_mmap=reload_mmap)

    mmap = np.memmap(mmap_loc, dtype='float32', mode='r', shape=(cutoff_shape + 2, glove_dim))

    test_len = test_df.shape[0]

    i = 0
    test_batch = 10000

    with tf.Session(graph=gr,config=tf.ConfigProto(log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restore model and continue
        ckpt = tf.train.get_checkpoint_state(chkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        while (i <= test_len):

            if (i + test_batch > test_len):
                test_batch = test_len - i
            else:
                test_batch = 10000

            test_sample = meta_data[i:i+test_batch]

            # Y_len_2 is just 0s, needed to pump into the graph. Unused otherwise
            qn_npy, qn_batch_len, sentence_len, sentence_batch_train_2, y_len_2 = process_questions(test_sample, glove_dict,
                                                                                                mmap,is_training=False)
            ylen2_npy = np.asarray(y_len_2)

            # Create a matrix with each row as sentence offsets
            # That gets used to rollup the flattened sentences into their documents
            sentence_offsets = np.cumsum(sentence_len)
            sentence_offsets_2 = np.insert(sentence_offsets, 0, 0, axis=0)
            sentence_offsets_3 = np.delete(sentence_offsets_2, sentence_offsets_2.shape[0] - 1)
            np_offsets_len = np.column_stack([sentence_offsets_3, sentence_offsets])

            prob, y_2, pred_indices = \
                sess.run([final_probs, tf_y_final_2, predicted_indices], feed_dict={
                    inputs: qn_npy,
                    batch_sequence_lengths: qn_batch_len,
                    sentence_batch_len: sentence_len,
                    sentence_index_offsets: np_offsets_len,
                    sentence_batch_length_2: sentence_batch_train_2,
                    ylen_2: ylen2_npy,
                    keep_fc: 1.0,
                    keep_conv : 1.0
                })

            y_2 = np.expand_dims(y_2,axis=1)
            pred_indices = np.expand_dims(pred_indices,axis=1)

            pred_indices = pred_indices.astype(int)
            y_2 = y_2.astype(int)

            batch_hstack = np.hstack((y_2,pred_indices))

            batch_final = np.concatenate([batch_final,batch_hstack],axis=0)

            i += test_batch

        batch_final = batch_final[1:,:]

        batch_df = pd.DataFrame(batch_final)
        batch_df.columns = ['qid_num','prediction']

        batch_merge = batch_df.merge(test_df,on='qid_num')
        batch_merge.drop('qid_num',axis=1,inplace=True)
        batch_merge.drop('question_text', axis=1, inplace=True)

        cols = ['qid','prediction']
        batch_merge = batch_merge.reindex(columns=cols)
        batch_merge[['prediction']] = batch_merge[['prediction']].astype(int)

        batch_merge.to_csv(out_file,index=False)



def check_validation_labels(valid_file, glove_file, mmap_loc, chkpoint_dir, out_file):

    """
    This is a copy of the inference function, but on a dev dataset. Useful to see the results, what the model is doing.

    :param valid_file: sample dev data file
    :param glove_file: glove embeddings
    :param mmap_loc: memory map location on disk
    :param chkpoint_dir: checkpoint directory
    :param out_file: file to which predictions are written.
    :return: None
    """

    reload_mmap = False  # Inference

    batch_final = np.zeros((1, 2))
    test_df = pd.read_csv(valid_file)

    test_df['qid_num'] = range(0, len(test_df))

    meta_data = test_df[['qid_num', 'question_text']]
    print(meta_data.shape)
    print(meta_data.head(10))

    threshold = 0.5

    # Build the graph and the optimizer and loss
    with tf.Graph().as_default() as gr:
        final_probs, logits, inputs, batch_sequence_lengths, sentence_batch_len, \
        sentence_index_offsets, sentence_batch_length_2, tf_y_final_2, ylen_2, \
        learning_rate_input, train_step, confusion_matrix, cross_entropy_mean, \
        loss, global_step, predicted_indices, keep_fc, keep_conv = \
            build_graph(cutoff_seq)

    _, _, glove_dict = get_train_df_glove_dict(None, glove_file, mmap_loc, is_training=False,
                                               reload_mmap=reload_mmap)

    mmap = np.memmap(mmap_loc, dtype='float32', mode='r', shape=(cutoff_shape + 2, glove_dim))

    test_len = test_df.shape[0]

    i = 0
    test_batch = 10000

    with tf.Session(graph=gr, config=tf.ConfigProto(log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restore model and continue
        ckpt = tf.train.get_checkpoint_state(chkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        while (i <= test_len):

            if (i + test_batch > test_len):
                test_batch = test_len - i
            else:
                test_batch = 10000

            test_sample = meta_data[i:i + test_batch]

            # Y_len_2 is just 0s, needed to pump into the graph. Unused otherwise
            qn_npy, qn_batch_len, sentence_len, sentence_batch_train_2, y_len_2 = process_questions(test_sample,
                                                                                                    glove_dict,
                                                                                                    mmap,
                                                                                                    is_training=False)
            ylen2_npy = np.asarray(y_len_2)

            # Create a matrix with each row as sentence offsets
            # That gets used to rollup the flattened sentences into their documents
            sentence_offsets = np.cumsum(sentence_len)
            sentence_offsets_2 = np.insert(sentence_offsets, 0, 0, axis=0)
            sentence_offsets_3 = np.delete(sentence_offsets_2, sentence_offsets_2.shape[0] - 1)
            np_offsets_len = np.column_stack([sentence_offsets_3, sentence_offsets])

            prob, y_2, pred_indices = \
                sess.run([final_probs, tf_y_final_2, predicted_indices], feed_dict={
                    inputs: qn_npy,
                    batch_sequence_lengths: qn_batch_len,
                    sentence_batch_len: sentence_len,
                    sentence_index_offsets: np_offsets_len,
                    sentence_batch_length_2: sentence_batch_train_2,
                    ylen_2: ylen2_npy,
                    keep_fc: 1.0,
                    keep_conv : 1.0
                })

            y_2 = np.expand_dims(y_2, axis=1)
            pred_indices = np.expand_dims(pred_indices, axis=1)

            pred_indices = pred_indices.astype(int)
            y_2 = y_2.astype(int)

            batch_hstack = np.hstack((y_2, pred_indices))

            batch_final = np.concatenate([batch_final, batch_hstack], axis=0)

            i += test_batch

        batch_final = batch_final[1:, :]

        batch_df = pd.DataFrame(batch_final)
        batch_df.columns = ['qid_num', 'prediction']

        batch_merge = batch_df.merge(test_df, on='qid_num')
        batch_merge.drop('qid_num', axis=1, inplace=True)
        batch_merge[['prediction']] = batch_merge[['prediction']].astype(int)

        batch_merge.to_csv(out_file, index=False)


def main():

    """
    Change parameters here e.g file locations
    TODO: config for hyperparameters through command line
    To search for hyperparameters in this file, I have commented wih 'hyperparameter'. Search for this string.

    :return: Nothing
    """

    chkpoint_dir = '/home/nitin/Desktop/kaggle_data/all/tensorboard/checkpoint/'
    #chkpoint_dir = '/kaggle/working/checkpoint/'

    out_file = '/home/nitin/Desktop/kaggle_data/all/test_inf.csv'
    out_valid = '/home/nitin/Desktop/kaggle_data/all/out_valid.csv'
    #out_file = 'submissions.csv'

    memmap_loc = '/home/nitin/Desktop/kaggle_data/all/memmap_file_embeddings.npy'
    #memmap_loc = 'memmap_file_embeddings.npy'

    glove_file = '/home/nitin/Desktop/kaggle_data/all/embeddings/glove.840B.300d/glove.840B.300d.txt'

    train_file = '/home/nitin/Desktop/kaggle_data/all/train.csv'
    test_data = '/home/nitin/Desktop/kaggle_data/all/test.csv'
    valid_dump = '/home/nitin/Desktop/kaggle_data/all/valid_dump.csv'

    train_tensorboard_dir = '/home/nitin/Desktop/kaggle_data/all/tensorboard/train/'
    valid_tensorboard_dir = '/home/nitin/Desktop/kaggle_data/all/tensorboard/valid/'

    #train_tensorboard_dir = '/kaggle/working/train/'
    #valid_tensorboard_dir = '/kaggle/working/valid/'

    build_session(train_file,glove_file,memmap_loc,chkpoint_dir,train_tensorboard_dir,valid_tensorboard_dir)
    check_validation_labels(valid_dump,glove_file,memmap_loc,chkpoint_dir,out_valid)
    #inference(test_data,glove_file,memmap_loc,chkpoint_dir,out_file)


main()








