#!/usr/bin/env python3


from Bernoulli_NB import Bernoulli_NaiveBayes_Classifier

from sklearn.model_selection import train_test_split


import numpy as np


from sklearn import metrics


import csv



import time

import logging
import logging.config

import json


import re


#b_partial = True
b_partial = False


b_cleanup = True


b_wo_sw = False

b_w_lemmatize = True

b_w_stemming = False


b_w_nwn = True

b_wo_punctuation = False



b_sentiment_words_filter = True


b_negate = b_sentiment_words_filter


#b_negate = True

b_separate_wp = True



i_partial_count = 1000


b_do_model_selection = False
#b_do_model_selection = True


logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


str_fn_postfix = ""



if b_cleanup:
    str_fn_postfix += "_cleanup"


if b_wo_sw:
    str_fn_postfix += "_wo_sw"

if b_negate:
    str_fn_postfix += "_negate"

if b_wo_punctuation:
    str_fn_postfix += "_wo_punctuation"

if b_w_lemmatize:
    str_fn_postfix += "_w_lemmatize"
    

if b_w_stemming:
    str_fn_postfix += "_w_stemming"


if b_w_nwn:
    str_fn_postfix += "_w_nwn"
    
    
if b_sentiment_words_filter:
    str_fn_postfix += "_w_swf"

if b_separate_wp:
    str_fn_postfix += "_w_separate_wp"



if True:
    str_fn_postfix += "_stat"


#str_fn_postfix += "_try"


b_use_original_text = True


str_fn_postfix = "_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified"


str_json_fn_training = "../" + "training" + "_set" + str_fn_postfix + ".json"

str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"


logger.info("str_json_fn_training = %s \n", str_json_fn_training)

logger.info("str_json_fn_testing = %s \n", str_json_fn_testing)

logger.info("begin pre-process for training datas... ")

X_list_raw = []
Y_list_raw = []

all_files_list_raw = []



jsonFile = open(str_json_fn_training, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_training_data_set = []

if not b_partial:
    real_training_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    
    real_training_data_set = sorted(json_data, key=lambda x: (x['id']), reverse=False)    
    real_training_data_set = [item for item in real_training_data_set[:i_partial_count] if item['y_val'] > -2]
    
    
real_training_data_set_sorted = sorted(real_training_data_set, key=lambda x: (x['id']), reverse=False)


logger.info("len of real_training_data_set_sorted = %d ", len(real_training_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
tmp_files_list_raw = []
tmp_pure_files_list_raw = []


for data_point in real_training_data_set_sorted:
    
    #tmp_list_X_raw.append(data_point['text'])
	
    i_se = data_point['i_sentiment_estimate']
    
    str_st = data_point['text']
    
    if b_use_original_text:
        #str_st = data_point['text_simple_cleanup']
        str_st = data_point['text_letter_and_num']
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass
    
    tmp_list_X_raw.append(str_st)	
    tmp_list_Y_raw.append(data_point['y_val'])	
    
	
    tmp_files_list_raw.append(data_point['full_file_name'])	
    tmp_pure_files_list_raw.append(data_point['id'])

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)


logger.info("end pre-process for training datas... ")





logger.info("begin pre-process for testing datas... ")


X_list_real_test_raw = []
Y_list_real_test_raw = []

all_files_list_real_test_raw = []
all_pure_files_list_real_test_raw = []



jsonFile = open(str_json_fn_testing, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_testing_data_set = []


if not b_partial:
    real_testing_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    real_testing_data_set = [item for item in json_data[:i_partial_count] if item['y_val'] > -2]


real_testing_data_set_sorted = sorted(real_testing_data_set, key=lambda x: (x['id']), reverse=False)


logger.info("len of real_testing_data_set_sorted = %d ", len(real_testing_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
tmp_files_list_raw = []
tmp_pure_files_list_raw = []


for data_point in real_testing_data_set_sorted:
    
    #tmp_list_X_raw.append(data_point['text'])
    
    #tmp_list_X_raw.append(data_point['senti_text'])
    
    
    i_se = data_point['i_sentiment_estimate']
    
    str_st = data_point['text']   
    if b_use_original_text:
        #str_st = data_point['text']
        #str_st = data_point['text_simple_cleanup']
        str_st = data_point['text_letter_and_num']
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass
    
    tmp_list_X_raw.append(str_st)

    tmp_list_Y_raw.append(data_point['y_val'])	
	
    tmp_files_list_raw.append(data_point['full_file_name'])	
    tmp_pure_files_list_raw.append(data_point['id'])

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)


logger.info("end pre-process for testing datas... ")






# seems that the all data (both training and testing) are combined together and then being splited, 
# according to train_size and test_size parameters of train_test_split
#X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.6, test_size=0.4)


logger.info("start train_test_split. ")


i_random_state=20

X_train, X_test, Y_train, Y_test, all_files_train, all_files_test = train_test_split(X_list_raw, Y_list_raw, all_files_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)


logger.info("end train_test_split. ")

print("len of X_train, X_test, Y_train, Y_test, all_files_train, all_files_test \n", len(X_train), len(X_test), len(Y_train), len(Y_test), len(all_files_train), len(all_files_test))

i_random_index = 15

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_train: ==============================================================\n")
print(X_train[i_random_index])

print("\n\n\n\n item of Y_train: ==============================================================\n")
print(Y_train[i_random_index])

print("\n\n\n\n item of all_files_train: ==============================================================\n")
print(all_files_train[i_random_index])




i_random_index = 18

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_test: ==============================================================\n")
print(X_test[i_random_index])

print("\n\n\n\n item of Y_test: ==============================================================\n")
print(Y_test[i_random_index])

print("\n\n\n\n item of all_files_test: ==============================================================\n")
print(all_files_test[i_random_index])





i_random_index = 20

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_list_real_test_raw: ==============================================================\n")
print(X_list_real_test_raw[i_random_index])

print("\n\n\n\n item of Y_list_real_test_raw: ==============================================================\n")
print(Y_list_real_test_raw[i_random_index])

print("\n\n\n\n item of all_files_list_real_test_raw: ==============================================================\n")
print(all_files_list_real_test_raw[i_random_index])





X_train_whole = []

X_train_whole.extend(X_train)
X_train_whole.extend(X_test)

Y_train_whole = []

Y_train_whole.extend(Y_train)
Y_train_whole.extend(Y_test)


all_files_whole_train = []

all_files_whole_train.extend(all_files_train) 
all_files_whole_train.extend(all_files_test)






print("len of X_train_whole and Y_train_whole", len(X_train_whole), len(Y_train_whole))


print("len of X_train and Y_train", len(X_train), len(Y_train))




bnb = Bernoulli_NaiveBayes_Classifier()


logger.info("start fit for pclf")

    
    
#pclf.fit(X_train_whole[0:200], Y_train_whole[0:200])

#pclf.fit(X_train_whole, Y_train_whole)

#pclf.fit(X_train, Y_train)

bnb.fit(X_train, Y_train)



logger.info("end fit for pclf")


#tmp_vect_X = pclf.named_steps['vect'].X


#print("\n\n\n\n tfidf idf_", tmp_vect_X, len(tmp_vect_X), type(tmp_vect_X), tmp_vect_X.shape)


print("Number of features found: ", len(bnb.features))






logger.info("start predict for X_test ")

#Y_test_pred = model_search_LR.predict(X_test)

#Y_test_cv_pred = cross_val_predict(pclf, X_test, Y_test, cv=5, n_jobs=2)

Y_test_pred = []


#Y_test_pred = pclf.predict(X_test)



Y_test_pred = bnb.predict(X_test)

print(metrics.classification_report(Y_test, Y_test_pred, digits=5))



Y_test_cv_pred = []
Y_test_cv_pred.extend([0] * int(len(Y_test) / 2))
Y_test_cv_pred.extend([1] * int(len(Y_test) / 2))

print("\n\n type and len of Y_test_cv_pred: ", type(Y_test_cv_pred), len(Y_test_cv_pred))


logger.info("end predict for X_test")




print("\n\n\n\n")



logger.info("start metrics.classification_report ")

print("\n\nmetrics.classification_report for Y_test and Y_test_pred \n")

print(metrics.classification_report(Y_test, Y_test_pred, digits=5))


print("\n\nmetrics.classification_report for Y_test and Y_test_cv_pred \n")

print(metrics.classification_report(Y_test, Y_test_cv_pred, digits=5))

logger.info("end metrics.classification_report ")





logger.info("start predict for X_list_real_test_raw ")

#Y_real_test_pred = model_search_LR.predict(X_list_real_test_raw)
Y_real_test_pred = []


Y_real_test_pred = bnb.predict(X_list_real_test_raw)

logger.info("end predict for X_list_real_test_raw")



print("\n\n\n\n")

#print("Y_test = \n\n", Y_test, type(Y_test))

print("\n\n\n\n")



#list_Y_test_pred = Y_test_pred.tolist()

#print("list_Y_test_pred = \n\n", list_Y_test_pred, type(list_Y_test_pred))

print("\n\n\n\n")


#Y_test_cv_pred

#list_Y_test_cv_pred = Y_test_cv_pred.tolist()

#print("list_Y_test_cv_pred = \n\n", list_Y_test_cv_pred, type(list_Y_test_cv_pred))




print("\n\n\n\n")

#list_Y_real_test_pred = Y_real_test_pred.tolist()
list_Y_real_test_pred = Y_real_test_pred


#print("list_Y_real_test_pred = \n\n", list_Y_real_test_pred, type(list_Y_real_test_pred))


print("\n\n\n\n")


print("len of list_Y_real_test_pred: ", len(list_Y_real_test_pred))



print("\n\n\n\n")





cur_time = int(time.time())

print("cur_time = ", cur_time)

str_file_name_tw = "../csv/group12_" + "bernoulli_nb" + "_" + str(cur_time) + "_submission" + str_fn_postfix + ".csv"


csv_stat_all_data = []

i_index = 0

for item_predict in list_Y_real_test_pred:
    
    csv_stat_item = {}
    
    tmp_pure_file_name = all_pure_files_list_real_test_raw[i_index]
    wlist = tmp_pure_file_name.split('.')
    
    csv_stat_item['id'] = int(wlist[0])

    csv_stat_item['category'] = item_predict
    #csv_stat_item['category'] = item_predict    
    
    i_index += 1
     
    csv_stat_all_data.append(csv_stat_item)



csv_set_sorted = sorted(csv_stat_all_data, key=lambda x: (x['id']), reverse=False)


with open(str_file_name_tw, 'wt', newline='') as csvfile:
    
    csv_writer = csv.writer(csvfile, dialect='excel')
    
    
    header = ['Id', 'Category']
    csv_writer.writerow(header)
    
    for data_point in csv_set_sorted:
        
        row = [data_point['id'], data_point['category']]
        #str_to_write = "\n" + str(data_point['id']) + "," + str(data_point['category'])
        #fout_tw.write(str_to_write)
        csv_writer.writerow(row)
        
    


logger.info("program ends. ")





