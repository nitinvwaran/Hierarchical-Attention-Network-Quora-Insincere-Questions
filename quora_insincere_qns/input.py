import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import pandas as pd
import re
import numpy as np

#train_1 = train_df.loc[train_df['target'] == 1]
#test_1 = test_df.loc[test_df['target'] == 1]

#print (train_df.head(10))
#print (train_df.shape)
#print (train_1.shape)


glove_dict = {}

def load_glove_vectors(file):

    temp_cache = '/home/nitin/Desktop/kaggle_data/all/temp.csv'

    with open(temp_cache,'w') as tmp:
        with open (file,'r') as f:
            for index, line in enumerate(f):
                l = line.split(' ')
                tmp.write(str(index) + ',' + str(l[0] + '\n'))

                #add to the global dictionary
                glove_dict[str(l[0]).strip().lower()] = index

    return glove_dict


def read_train_test_words(train_file,test_file, glove_file):

    UNK = 2196017

    qn_idxs = []

    glove_dict = load_glove_vectors(glove_file)
    train_df = pd.read_csv(train_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)
    #train_qns = train_df.iloc[:,1]

    for index, item in train_df.iterrows():

        qn = item[1].split(' ')
        qn_ls = [re.sub('[^A-Za-z0-9]+', '', q) for q in qn]
        qn_ls = [x.lower() for x in qn_ls]# if x not in stop_words]
        print (qn_ls)
        qn_idx = [glove_dict[x] if (x in glove_dict.keys()) else UNK for x in qn_ls]
        qn_idxs.append(qn_idx)

    qn_npy = np.asarray(qn_idxs)

    print (qn_npy.shape)







#def build_graph():



def main():
    glove_vectors_file = '/home/nitin/Desktop/kaggle_data/all/embeddings/glove.840B.300d/glove.840B.300d.txt'
    train_data = '/home/nitin/Desktop/kaggle_data/all/train.csv'
    test_data = '/home/nitin/Desktop/kaggle_data/all/test.csv'

    #load_glove_vectors(glove_vectors_file)
    read_train_test_words(train_data,test_data,glove_vectors_file)


main()








