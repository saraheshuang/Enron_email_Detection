from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
import numpy as np
import pandas as pd

def split_data(my_dataset, feature_list, random_state):
    data = featureFormat(my_dataset, feature_list, sort_keys = True)
    labels, features= targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=random_state)
    features_train=np.array(features_train)
    features_test=np.array(features_test)
    labels_train=np.array(labels_train)
    labels_test=np.array(labels_test)
    return features_train, features_test, labels_train, labels_test

def features_importance(features_train, labels_train, feature_list):   
    X=SelectKBest(k=5)
    X.fit(features_train, labels_train)
    Scores=X.scores_
    Pvalues=X.pvalues_
    index=feature_list[1:]
    return pd.DataFrame({'Scores': Scores,'Pvalues': Pvalues},index=index)

def imputed_data(features):
    imr = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(features)
    newdata= imr.transform(features)
    return np.array(newdata)

def find_best_k(my_dataset, feature_list, random_states):
    final_scores={}
    final_pvalue={}
    for i in random_states:
        features_train, features_test, labels_train, labels_test=split_data(my_dataset, feature_list, i)
        features_train=imputed_data(features_train)
        table_importance=features_importance(features_train, labels_train, feature_list)
        for feature in feature_list[1:]:
            try:
                final_scores[feature]+=table_importance['Scores'][feature]/200
                final_pvalue[feature]+=table_importance['Pvalues'][feature]/200
            except KeyError:
                final_scores[feature]=table_importance['Scores'][feature]/200
                final_pvalue[feature]=table_importance['Pvalues'][feature]/200
    return pd.DataFrame({'Pvalues': final_pvalue, 'Scores': final_scores}, index=feature_list[1:]).sort_values('Pvalues')