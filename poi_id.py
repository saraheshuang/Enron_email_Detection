#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
 # You will need to use more features
#

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
enron_data=pd.DataFrame(data_dict).T
##remove the outlier

salary_outlier=max(enron_data['salary'])

def find_outlier(dataset, variable_name, value):
    observation_name=''
    for i in range(len(dataset.keys())):
        if dataset[dataset.keys()[i]][variable_name]==value:
            observation_name=dataset.keys()[i]
    return observation_name

outlier_name=find_outlier(data_dict, 'salary', salary_outlier)
data_dict.pop(outlier_name, 0)


##create new features call poi_email_porpotion
for i in data_dict:
    data_dict[i]['poi_email_propotion_from']=float(data_dict[i]['from_poi_to_this_person'])/float(data_dict[i]['to_messages'])
    data_dict[i]['poi_email_propotion_to']=float(data_dict[i]['from_this_person_to_poi'])/float(data_dict[i]['from_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict


features_list=['poi', 'salary',  'total_payments', 'exercised_stock_options', 'bonus', 
'restricted_stock', 'shared_receipt_with_poi', 'poi_email_propotion_to',
'total_stock_value', 'expenses', 'deferred_income', 
'long_term_incentive', 'from_poi_to_this_person']
##Please refer the ipython notebook to see how this feature_list is generated

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.




#SVM line
#decision tree
#ensemble 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#GNB Model
steps_gnb = [('pre_pro', Imputer(missing_values='NaN', strategy='mean', axis=0)),
             ('feature_scaling', StandardScaler()),
                ('reduce_dim', PCA()), 
                ('GNB', GaussianNB())]
GNB = Pipeline(steps_gnb)
cv=StratifiedShuffleSplit(labels, 1000)
params_gnb = dict( reduce_dim__n_components=[5, 6, 7, 8, 9, 10])
clf_gnb = GridSearchCV(GNB, param_grid=params_gnb, cv=cv, scoring='f1')
clf_gnb.fit(features, labels)
clf_gnb=clf_gnb.best_estimator_

#RomdonForest
steps_rf = [('pre_pro', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                 ('rf', RandomForestClassifier())]
rf = Pipeline(steps_rf)
cv=StratifiedShuffleSplit(labels, 200)
params_rf = dict(sek__k=[8,9,10], reduce_dim__n_components=[5,6,7,8])
clf_rf = GridSearchCV(rf, param_grid=params_rf, scoring='f1', cv=cv)
clf_rf.fit(features, labels)
clf_rf=clf_rf.best_estimator_
#SVC
steps_svc = [('pre_pro', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                ('feature_scaling', StandardScaler()),     
                 ('reduce_dim', PCA()), 
                 ('svm', SVC(kernel='linear'))]
svc = Pipeline(steps_svc)
cv=StratifiedShuffleSplit(labels, 200)
params_svc = dict(reduce_dim__n_components=[5,6,7,8,9], svm__C=[1, 10, 100])
clf_svc = GridSearchCV(svc, param_grid=params_svc, cv=cv, scoring='recall')
clf_svc.fit(features, labels)
clf_svc=clf_svc.best_estimator_
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf=clf_gnb

dump_classifier_and_data(clf, my_dataset, features_list)