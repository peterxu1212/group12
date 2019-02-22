from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import numpy as np

from sklearn.exceptions import FitFailedWarning
import warnings

from sklearn.pipeline import Pipeline

#from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

#from sklearn.svm import SVC

from sklearn.svm import LinearSVC

import csv



import time

import logging
import logging.config

import json

#from os import listdir
#from os.path import isfile, join

#from pathlib import Path

import re

from bs4 import BeautifulSoup

import time

import matplotlib.pyplot as plt

str_fn_postfix = ""

str_fn_postfix += "_cleanup"

str_fn_postfix += "_w_lemmatize"

str_json_fn_training = "../" + "training" + "_set" + str_fn_postfix + ".json"

str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"

X_list_raw = []
Y_list_raw = []

all_files_list_raw = []

jsonFile = open(str_json_fn_training, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file

real_training_data_set = []

real_training_data_set = [item for item in json_data if item['y_val'] > -2]
real_training_data_set_sorted = sorted(real_training_data_set, key=lambda x: (x['id']), reverse=False)

tmp_list_X_raw = []
tmp_list_Y_raw = []
tmp_files_list_raw = []
tmp_pure_files_list_raw = []

for data_point in real_training_data_set_sorted:

	tmp_list_X_raw.append(data_point['text'])
	tmp_list_Y_raw.append(data_point['y_val'])

	tmp_files_list_raw.append(data_point['full_file_name'])
	tmp_pure_files_list_raw.append(data_point['id'])

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)

i_random_state=20

X_train, X_test, Y_train, Y_test, all_files_train, all_files_test = train_test_split(X_list_raw, Y_list_raw, all_files_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)

def ppl_score(set_min_df=0.0, set_max_df=1.0, set_max_features=None, set_ngram_range_max=4, set_C=1.0, set_max_iter=5000):
    pclf = Pipeline([
        ('vect', CountVectorizer(min_df=set_min_df, max_df=set_max_df, max_features=set_max_features, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
        ('norm', Normalizer()),
        ('clf', LinearSVC(C=set_C, random_state=10, tol=1e-05, max_iter=set_max_iter)),
    ])

    scores = cross_val_score(pclf, X_test, Y_test, cv=5)
    scores_mean = scores.mean()
    return scores_mean

#Firstly, plot the affect of set_min_df
list_min_df = []
list_score = []
list_time = []

for set_min_df in range (0, 12, 2):
    list_min_df.append(set_min_df)
    start_time = time.time()

    score = ppl_score(0.0001*set_min_df, 1.0, None, 4, 1.0, 5000)

    elapsed_time = time.time() - start_time
    list_time.append(elapsed_time)
    list_score.append(score)

plt.plot(list_min_df, list_score, "bo")
plt.xlabel("Min df")
plt.ylabel("Score")
plt.title("Plot of Score w.r.t Min df")
plt.show()

plt.plot(list_min_df, list_time, "bo")
plt.xlabel("Min df")
plt.ylabel("Time")
plt.title("Plot of Time w.r.t Min df")
plt.show()

#Secondly, plot the affect of max_df
list_max_df = []
list_score = []
list_time = []
for set_max_df in range (90, 102, 2):
    list_max_df.append(set_max_df)
    start_time = time.time()

    score = ppl_score(0.0006, 0.01*set_max_df, None, 4, 1.0, 5000)

    elapsed_time = time.time() - start_time
    list_time.append(elapsed_time)
    list_score.append(score)

plt.plot(list_max_df, list_score, "bo")
plt.xlabel("Max df")
plt.ylabel("Score")
plt.title("Plot of Score w.r.t Max df")
plt.show()

plt.plot(list_max_df, list_time, "bo")
plt.xlabel("Max df")
plt.ylabel("Time")
plt.title("Plot of Time w.r.t Max df")
plt.show()

#Thirdly, plot the affect of C
list_C = []
list_score = []
list_time = []
for set_C in range (20, 110, 20):
    list_C.append(set_C)
    start_time = time.time()

    score = ppl_score(0.0006, 0.96, None, 4, set_C, 5000)

    elapsed_time = time.time() - start_time
    list_time.append(elapsed_time)
    list_score.append(score)

plt.plot(list_C, list_score, "bo")
plt.xlabel("C")
plt.ylabel("Score")
plt.title("Plot of Score w.r.t C")
plt.show()

plt.plot(list_C, list_time, "bo")
plt.xlabel("C")
plt.ylabel("Time")
plt.title("Plot of Time w.r.t C")
plt.show()
