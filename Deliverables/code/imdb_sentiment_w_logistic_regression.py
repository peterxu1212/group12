#!/usr/bin/env python3





from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import GridSearchCV

import numpy as np

from sklearn.exceptions import FitFailedWarning
import warnings


#from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

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

def pre_process_data(in_str_file_path, in_int_y_val, in_file_name_list = [], b_guess_when_exception=False, b_test=False):
    
    out_X_list_raw = []
    out_Y_list_raw = []
    
    out_files_list_raw = []
    out_pure_files_list_raw = []
        
    #for str_file_name in in_file_name_list[:100]:
    for str_file_name in in_file_name_list:
    
        #print(str_file_name, type(str_file_name))
        str_full_file_name = in_str_file_path + str_file_name
        file_content = ""
    
        try:
            
            file_handle = open(str_full_file_name, encoding = 'gbk',errors = 'ignore')
            #file_handle = open(str_full_file_name, encoding = 'gb18030',errors = 'ignore')
            #file_handle = open(str_full_file_name, encoding = 'utf-8',errors = 'ignore')
            
            
            #file = open(path, encoding='gb18030'）
            
            
            
            #file_content = Path(str_full_file_name).read_text()
            file_content = file_handle.read()
            
            tmp_soup = BeautifulSoup(file_content, 'html.parser')
            
            file_content = tmp_soup.get_text()
            
            file_content = re.sub("[^a-zA-Z]", " ", file_content)
            
            #if b_wo_punctuation:
                #file_content = re.sub(r'[^\w\s]', ' ', file_content)
                
            file_content = file_content.lower()
            
            #if str_file_name == "10327_7.txt":
            #    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
            #    print("\n\n\n\n\n\n file_content = ", file_content)
            
            
            if b_test:
                file_content = "bad"
            
        except Exception as e:
            str_tmp = "pre_process_data_files Exception e = " + str(e) + " when execute read_text for str_full_file_name = \n\n" + str(str_full_file_name) +  "\n\n"
        
            logger.warning("%s", str_tmp)
            #print(str_tmp)
            
            if not b_guess_when_exception:
                continue
            else:
                file_content = "bad"
                
    
        out_X_list_raw.append(file_content)
        out_Y_list_raw.append(in_int_y_val)
        out_files_list_raw.append(str_full_file_name)
        out_pure_files_list_raw.append(str_file_name)
    
    return out_X_list_raw, out_Y_list_raw, out_files_list_raw, out_pure_files_list_raw



#b_partial = True
    
b_partial = False

b_wo_punctuation = True


i_partial_count = 500




logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")




str_json_fn_training = "../" + "training" + "_set.json"

str_json_fn_testing = "../" + "testing" + "_set.json"




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
	
	tmp_list_X_raw.append(data_point['text'])
	tmp_list_Y_raw.append(data_point['y_val'])
	
	tmp_files_list_raw.append(data_point['full_file_name'])	
	tmp_pure_files_list_raw.append(data_point['id'])

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)


logger.info("end pre-process for training datas... ")


"""
logger.info("begin pre-process for pos training datas... ")


tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, _ = pre_process_data_files(str_train_files_pos_path, 1, train_files_pos)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for pos training datas... ")




logger.info("begin pre-process for neg training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, _ = pre_process_data_files(str_train_files_neg_path, 0, train_files_neg)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for neg training datas... ")
"""   

#print("X_list_raw = \n", X_list_raw)

#print("Y_list_raw = \n", Y_list_raw)






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
	
	tmp_list_X_raw.append(data_point['text'])
	tmp_list_Y_raw.append(data_point['y_val'])
	
	tmp_files_list_raw.append(data_point['full_file_name'])	
	tmp_pure_files_list_raw.append(data_point['id'])

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)


logger.info("end pre-process for testing datas... ")



"""
logger.info("begin pre-process for pos testing datas... ")


tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw = pre_process_data_files(str_test_files_path, 0, test_files, True, False)

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)
    
logger.info("end pre-process for pos testing datas... ")
"""



















# seems that the all data (both training and testing) are combined together and then being splited, 
# according to train_size and test_size parameters of train_test_split
#X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.6, test_size=0.4)


logger.info("start train_test_split. ")

X_train, X_test, Y_train, Y_test, all_files_train, all_files_test = train_test_split(X_list_raw, Y_list_raw, all_files_list_raw, train_size=0.8, test_size=0.2, random_state=12)


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



"""
tfv = TFIV(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
"""











vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')


#vectorizer = CountVectorizer()


logger.info("start vectorizer.fit ")


#count_vect = vectorizer.fit(X_train)
tfidf_vect = vectorizer.fit(X_train)


logger.info("end vectorizer.fit ")




fns = vectorizer.get_feature_names()

logger.info("\n\n vectorizer get_feature_names: \n\n %s", str(fns))

#print("\n\n vectorizer get_feature_names: ", str(fns))

i_len_fns = len(fns)

print("\n\n\n\n i_len_fns = ", i_len_fns)





logger.info("start tfidf_vect.transform for X_train ")

X_train_tfidf = tfidf_vect.transform(X_train)

logger.info("end tfidf_vect.transform for X_train ")



logger.info("start tfidf_vect.transform for X_test ")

X_test_tfidf = tfidf_vect.transform(X_test)

logger.info("end tfidf_vect.transform for X_test ")

#print("\n\n\n\n X_train_counts toarray: \n\n", X_train_counts.toarray())

#print("\n\n\n\n X_test_counts toarray: \n\n", X_test_counts.toarray())



logger.info("start tfidf_vect.transform for X_list_real_test_raw ")

X_real_test_tfidf = tfidf_vect.transform(X_list_real_test_raw)

logger.info("end tfidf_vect.transform for X_list_real_test_raw ")




print("X_train_tfidf = \n\n", X_train_tfidf)

print("\n\n X_train_tfidf.shape = ", X_train_tfidf.shape)

print("\n\n\n\n")





#print(X_test_counts)


#print("\n\n\n\n")



"""
logger.info("start TfidfTransformer().fit ")

tfidf_transformer = TfidfTransformer().fit(X_train_counts)

logger.info("end TfidfTransformer().fit ")




logger.info("start tfidf_transformer.transform for X_train_counts")

X_train_tfidf = tfidf_transformer.transform(X_train_counts)

logger.info("end tfidf_transformer.transform for X_train_counts ")


logger.info("start tfidf_transformer.transform for X_test_counts")

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

logger.info("end tfidf_transformer.transform for X_test_counts ")



logger.info("start tfidf_transformer.transform for X_real_test_counts")

X_real_test_tfidf = tfidf_transformer.transform(X_real_test_counts)

logger.info("end tfidf_transformer.transform for X_real_test_counts ")



print("X_train_tfidf = \n\n", X_train_tfidf)


print("\n\n\n\n")
"""



X_train_normalized = X_train_tfidf
X_test_normalized = X_test_tfidf
X_real_test_normalized = X_real_test_tfidf

"""
logger.info("start Normalizer().fit ")

normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)

logger.info("end Normalizer().fit ")




logger.info("start normalizer_tranformer.transform for X_train_tfidf ")

X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)

logger.info("end normalizer_tranformer.transform for X_train_tfidf ")



logger.info("start normalizer_tranformer.transform for X_test_tfidf ")

X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

logger.info("end normalizer_tranformer.transform for X_test_tfidf ")




logger.info("start normalizer_tranformer.transform for X_real_test_tfidf ")

X_real_test_normalized = normalizer_tranformer.transform(X_real_test_tfidf)

logger.info("end normalizer_tranformer.transform for X_real_test_tfidf ")
"""


print("X_train_normalized = \n\n", X_train_normalized)


print("\n\n\n\n")

X_train_normalized_whole = []

X_train_normalized_whole.extend(X_train_normalized)
X_train_normalized_whole.extend(X_test_normalized)

Y_train_whole = []

Y_train_whole.extend(Y_train)
Y_train_whole.extend(Y_test)


all_files_whole_train = []

all_files_whole_train.extend(all_files_train) 
all_files_whole_train.extend(all_files_test)


print("len of X_train_normalized_whole and Y_train_whole", len(X_train_normalized_whole), len(Y_train_whole))

logger.info("start model_LR.fit ")

#grid_values = {'C':[1, 2, 4, 8, 16, 32]}

#params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(1,4)],
#           "C":[1, 4, 16, 32]
#}


params = {"C":[1, 4, 16, 32]
}


#clf = MultinomialNB().fit(X_train_normalized, Y_train)



model_LR = LogisticRegression(penalty = 'l2', dual = True, random_state = 0, solver='liblinear')
#model_LR = LogisticRegression(penalty = 'l2', dual = True, random_state = 0, solver='liblinear')

i_seed = 500

model_search_LR = GridSearchCV(model_LR, param_grid = params, scoring = 'roc_auc', cv = 10, verbose = 10, error_score='raise', iid=True)

with warnings.catch_warnings(record=True) as w:
    try:
        #gs.fit(X, y)   # This will raise a ValueError since C is < 0
        
        clf = model_search_LR.fit(X_train_normalized_whole, Y_train_whole) # Fit the model.
    except Exception as e:
        #except ValueError:
        print("\n\n Exception e = " + str(e))
        #pass
    print("\n\n w = \n", w)

# Try to set the scoring on what the contest is asking for. 
# The contest says scoring is for area under the ROC curve, so use this.

#error_score=np.nan


                        

print("\n\n model_LR.grid_scores_ = \n", model_LR.grid_scores_)

print("\n\n model_LR.best_estimator_ = \n", model_LR.best_estimator_)

print("\n\n model_LR.cv_results_ = \n", model_LR.cv_results_)

#clf = model_LR.fit(X_train_normalized, Y_train)


logger.info("end model_LR.fit ")




logger.info("start clf.predict for X_test_normalized ")

Y_test_pred = clf.predict(X_test_normalized)

logger.info("end clf.predict for X_test_normalized")



logger.info("start clf.predict for X_real_test_normalized ")

Y_real_test_pred = clf.predict(X_real_test_normalized)

logger.info("end clf.predict for X_real_test_normalized")



print("\n\n\n\n")

print("Y_test = \n\n", Y_test, type(Y_test))


print("\n\n\n\n")


list_Y_test_pred = Y_test_pred.tolist()

print("list_Y_test_pred = \n\n", list_Y_test_pred, type(list_Y_test_pred))


print("\n\n\n\n")


logger.info("start metrics.classification_report ")

print(metrics.classification_report(Y_test, Y_test_pred))


logger.info("end metrics.classification_report ")



print("\n\n\n\n")

list_Y_real_test_pred = Y_real_test_pred.tolist()


#print("list_Y_real_test_pred = \n\n", list_Y_real_test_pred, type(list_Y_real_test_pred))


print("\n\n\n\n")


print("len of list_Y_real_test_pred: ", len(list_Y_real_test_pred))



print("\n\n\n\n")




cur_time = int(time.time())

print("cur_time = ", cur_time)

str_file_name_tw = "../csv/group12_" + "logistic_regression" + "_" + str(cur_time) + "_submission.csv"


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



