'''
Name:		classifier.py 
Author:		Margaret Thomann & Michael McRoskey
Description:	File used to build a classifier for given data set
'''

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

TRAIN_FILE = "../data/classification_train.data"
TEST_FILE = "../data/classification_test.test"

# Read training data into pandas data frame
features_train = 	["f01", "f02", "f03", "f04", "f05", "f06", 
			"f07", "f08", "f09", "f10", "f11", "f12", 
			"f13", "f14", "f15", "f16", "f17", "f18", 
			"f19", "f20", "f21", "f22", "f23", "f24", 
			"f25", "f26", "f27", "f28", "f29", "f30", 
			"f31", "f32", "f33", "f34", "f35", "f36", 
			"f37", "f38", "f39", "f40", "f41", "f42", 
			"f43", "f44", "f45", "f46", "f47", "f48", 
			"Class"]
df_train = pd.read_csv(TRAIN_FILE, names=features_train)

# Read testing data into pandas data frame
features_test = 		["f01", "f02", "f03", "f04", "f05", "f06", 
			"f07", "f08", "f09", "f10", "f11", "f12", 
			"f13", "f14", "f15", "f16", "f17", "f18", 
			"f19", "f20", "f21", "f22", "f23", "f24", 
			"f25", "f26", "f27", "f28", "f29", "f30", 
			"f31", "f32", "f33", "f34", "f35", "f36", 
			"f37", "f38", "f39", "f40", "f41", "f42", 
			"f43", "f44", "f45", "f46", "f47", "f48"]
df_test = pd.read_csv(TRAIN_FILE, names=features_test)

# Define x and y from data
x_train = df_train.drop(['Class'], axis=1)
y_train = df_train['Class']


# Learn an sklearn k-fold cross validation classifier
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(x_train, y_train)
clf.predict(df_test)
results = []
cv_results = cross_val_score(clf, x_train, y_train, cv=10, scoring="accuracy")
cv_results_f_mlp = cross_val_score(clf, x_train, y_train, cv=10, scoring="f1")
results.append(cv_results)
print results[0:4]
results.append(cv_results_f_mlp)
