#!/usr/bin/env python3





from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import Normalizer

#from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.linear_model import LogisticRegression as LR

import csv



import time

import logging
import logging.config

from os import listdir
from os.path import isfile, join

from pathlib import Path


def pre_process_data_files(in_str_file_path, in_int_y_val, in_file_name_list = [], b_guess_when_exception=False, b_test=False):
    
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
            file_content = Path(str_full_file_name).read_text()
            
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



logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


str_test_files_path = "../../../comp-551-imbd-sentiment-classification/test/test/"


str_train_files_pos_path = "../../../comp-551-imbd-sentiment-classification/train/train/pos/"

str_train_files_neg_path = "../../../comp-551-imbd-sentiment-classification/train/train/neg/" 



logger.info("start enum %s folder. ", str_test_files_path)

test_files = [f for f in listdir(str_test_files_path) if isfile(join(str_test_files_path, f))]

logger.info("end enum %s folder. ", str_test_files_path)



logger.info("start enum %s folder. ", str_train_files_pos_path)

train_files_pos = [f for f in listdir(str_train_files_pos_path) if isfile(join(str_train_files_pos_path, f))]

logger.info("end enum %s folder. ", str_train_files_pos_path)



logger.info("start enum %s folder. ", str_train_files_neg_path)

train_files_neg = [f for f in listdir(str_train_files_neg_path) if isfile(join(str_train_files_neg_path, f))]

logger.info("end enum %s folder. ", str_train_files_neg_path)



i_test_file_count = len(test_files)
print("i_test_file_count = ", i_test_file_count)

print(test_files[10])



i_train_file_pos_count = len(train_files_pos)
print("i_train_file_pos_count = ", i_train_file_pos_count)

print(train_files_pos[10])




i_train_file_neg_count = len(train_files_neg)
print("i_train_file_neg_count = ", i_train_file_neg_count)

print(train_files_neg[10])




logger.info("after enum source datas \n")

#from sklearn.model_selection import cross_val_score


#newsgroups = fetch_20newsgroups(subset='all')


#print(newsgroups.target_names)

X_list_raw = []
Y_list_raw = []

all_files_list_raw = []


logger.info("begin pre-process for pos training datas... ")

#tmp_list_X_raw, tmp_list_Y_raw = pre_process_data_files(str_train_files_pos_path, 1, train_files_pos[:1000])

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, _ = pre_process_data_files(str_train_files_pos_path, 1, train_files_pos)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for pos training datas... ")




logger.info("begin pre-process for neg training datas... ")

#tmp_list_X_raw, tmp_list_Y_raw = pre_process_data_files(str_train_files_neg_path, 0, train_files_neg[:1000])
tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, _ = pre_process_data_files(str_train_files_neg_path, 0, train_files_neg)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for neg training datas... ")
    

#print("X_list_raw = \n", X_list_raw)

#print("Y_list_raw = \n", Y_list_raw)


X_list_real_test_raw = []
Y_list_real_test_raw = []

all_files_list_real_test_raw = []
all_pure_files_list_real_test_raw = []

logger.info("begin pre-process for pos testing datas... ")

#tmp_list_X_raw, tmp_list_Y_raw = pre_process_data_files(str_train_files_pos_path, 1, train_files_pos[:1000])

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw = pre_process_data_files(str_test_files_path, 0, test_files, True, False)

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)
    
logger.info("end pre-process for pos testing datas... ")




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







vectorizer = CountVectorizer()


logger.info("start vectorizer.fit ")


count_vect = vectorizer.fit(X_train)



logger.info("end vectorizer.fit ")


fns = vectorizer.get_feature_names()

logger.info("\n\n vectorizer get_feature_names: \n\n %s", str(fns))

i_len_fns = len(fns)

print("\n\n\n\n i_len_fns = ", i_len_fns)



logger.info("start count_vect.transform for X_train ")

X_train_counts = count_vect.transform(X_train)

logger.info("end count_vect.transform for X_train ")



logger.info("start count_vect.transform for X_test ")

X_test_counts = count_vect.transform(X_test)

logger.info("end count_vect.transform for X_test ")

#print("\n\n\n\n X_train_counts toarray: \n\n", X_train_counts.toarray())

#print("\n\n\n\n X_test_counts toarray: \n\n", X_test_counts.toarray())



logger.info("start count_vect.transform for X_list_real_test_raw ")

X_real_test_counts = count_vect.transform(X_list_real_test_raw)

logger.info("end count_vect.transform for X_list_real_test_raw ")




print("X_train_counts = \n\n", X_train_counts)


print("\n\n\n\n")


#print(X_test_counts)


#print("\n\n\n\n")




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



print("X_train_normalized = \n\n", X_train_normalized)


print("\n\n\n\n")





logger.info("start model_LR.fit ")

#clf = MultinomialNB().fit(X_train_normalized, Y_train)

model_LR = LR(penalty = 'l2', dual = True, random_state = 0, solver='liblinear')

clf = model_LR.fit(X_train_normalized, Y_train)


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
    
    #csv_writer.writerow(['Id'] + ['Category'])
    
    
    header = ['Id', 'Category']
    csv_writer.writerow(header)
    
    for data_point in csv_set_sorted:
        
        row = [data_point['id'], data_point['category']]
        #str_to_write = "\n" + str(data_point['id']) + "," + str(data_point['category'])
        #fout_tw.write(str_to_write)
        csv_writer.writerow(row)
        
    


logger.info("program ends. ")



