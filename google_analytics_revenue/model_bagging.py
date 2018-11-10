from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

def training():

    file_name = '/home/nitin/Desktop/google_analytics/google_analytics_revenue/dfgoog_mod1_v2_v2.csv'

    df_train = pd.read_csv(file_name)
    df_train.drop('Unnamed: 0',axis=1,inplace=True)
    y = df_train.loc[:, 'is_revenue']
    df_train.drop('is_revenue',axis=1,inplace=True)

    n_folds = 10
    auc = []

    rf = GradientBoostingClassifier(n_estimators=100)
    for i in range(0,n_folds):


        X_train, X_dev, y_train, y_dev = train_test_split(df_train,y,stratify=y,test_size=0.1)


        print ('Fold:' + str(i))
        print (X_train.shape)
        print (X_dev.shape)



        rf.fit(X_train,y_train)


        y_prob = rf.predict_proba(X_dev)[:,1]
        y_hat = (y_prob > 0.5)

        conf_matrix = confusion_matrix(y_dev,y_hat)
        print (conf_matrix)

        auc.append(roc_auc_score(y_dev,y_prob))
        print (auc[i])


    auc_avg = np.mean(auc)
    print ('The average auc ')
    print (auc_avg)

    #fit_model_after_cv(df_train,y)

def fit_model_after_cv(df_train, y):

    rf = RandomForestClassifier()

    rf.fit(df_train,y)
    probs = rf.predict_proba(df_train)[:,1]

    print ('Final AUC')
    print(roc_auc_score(y,probs))







training()




