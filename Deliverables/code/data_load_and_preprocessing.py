# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:00:31 2019

@author: PeterXu
"""



import logging
import logging.config


from os import listdir
from os.path import isfile, join

import json

from bs4 import BeautifulSoup

import re

import os


def pre_process_data_files(in_str_json_fn, in_str_file_path, in_int_y_val, in_file_name_list = [], b_guess_when_exception=False, b_test=False):
    
    pre_processed_data_set = []
    
    out_X_list_raw = []
    out_Y_list_raw = []
    
    out_files_list_raw = []
    out_pure_files_list_raw = []
    
    i_cur_processing_index = 0
        
    #for str_file_name in in_file_name_list[:100]:
    for str_file_name in in_file_name_list:
    
        i_cur_processing_index += 1
        
        if i_cur_processing_index % 500 == 0:
            logger.info("pre_process_data_files i_cur_processing_index = %d", i_cur_processing_index)
        

        str_full_file_name = in_str_file_path + str_file_name
        file_content = ""
        
        str_processed_text = ""
        
        pre_processed_item = {}
    
        try:
                        
            if not b_test:
                
                #file_handle = open(str_full_file_name, encoding = 'gbk', errors = 'ignore')
                #file_handle = open(str_full_file_name, encoding = 'gb18030', errors = 'ignore')
                file_handle = open(str_full_file_name, encoding = 'utf-8')

                file_content = file_handle.read()
                
                tmp_soup = BeautifulSoup(file_content, 'html.parser')
                
                file_content_wo_html_tags = tmp_soup.get_text()
                
                file_content_pure_english = re.sub("[^a-zA-Z]", " ", file_content_wo_html_tags)
                
                file_content_pure_english_w_1ws = ' '.join(file_content_pure_english.split())
                
                #if b_wo_punctuation:
                    #file_content = re.sub(r'[^\w\s]', ' ', file_content)
                    
                str_processed_text = file_content_pure_english_w_1ws.lower()
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n str_processed_text = ", str_processed_text)
            
                
            else:
                str_processed_text = "bad"
            
        except Exception as e:
            str_tmp = "pre_process_data_files Exception e = " + str(e) + " when execute read_text for str_full_file_name = \n\n" + str(str_full_file_name) +  "\n\n"
        
            logger.warning("%s", str_tmp)
            #print(str_tmp)
            
            if not b_guess_when_exception:
                continue
            else:
                str_processed_text = "bad"
                
        
           
        out_X_list_raw.append(str_processed_text)
        out_Y_list_raw.append(in_int_y_val)
        out_files_list_raw.append(str_full_file_name)
        out_pure_files_list_raw.append(str_file_name)
        
        pre_processed_item['text'] = str_processed_text
        pre_processed_item['y_val'] = in_int_y_val
        pre_processed_item['full_file_name'] = str_full_file_name
                
        wlist = str_file_name.split('.')    
        pre_processed_item['id'] = wlist[0]
        
        pre_processed_data_set.append(pre_processed_item)
        
        

    str_json_fn = in_str_json_fn
    
    jsonFile = open(str_json_fn, "r") # Open the JSON file for reading
    json_data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    logger.info("pre_process_data_files len of json_data = %d, for json file %s ", len(json_data), str_json_fn)

    ## Working with buffered content
    json_data.extend(pre_processed_data_set)
    
    logger.info("pre_process_data_files len of json_data = %d, after extend ", len(json_data))


    ## Save our changes to JSON file
    jsonFile = open(str_json_fn, "w")
    jsonFile.write(json.dumps(json_data))
    jsonFile.close()
    
    
    return out_X_list_raw, out_Y_list_raw, out_files_list_raw, out_pure_files_list_raw






"""
raw_text = '''
<td><a href="http://www.irit.fr/SC">Signal et Communication</a>
<br/><a href="http://www.irit.fr/IRT">Ingénierie Réseaux et Télécommunications</a>
</td>
'''

print("\n\n raw_text = \n", raw_text)




soup = BeautifulSoup(raw_text, 'html.parser')

#print("\n\n soup = \n", soup.prettify())

text_wo_html_tags = soup.get_text()


print("\n\n text_wo_html_tags = \n", text_wo_html_tags)

"""

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



logger.info("\n re-generate json files %s and %s \n\n", str_json_fn_training, str_json_fn_testing)


tmp_pre_processed_data_set = []

tmp_pre_processed_item = {}

tmp_pre_processed_item['text'] = ""
tmp_pre_processed_item['y_val'] = -2
tmp_pre_processed_item['full_file_name'] = ""
tmp_pre_processed_item['id'] = ""
        
tmp_pre_processed_data_set.append(tmp_pre_processed_item)


if os.path.isfile(str_json_fn_training):
    os.remove(str_json_fn_training)
else:    ## Show an error ##
    print(" %s file not found", str_json_fn_training)
    
tmp_json_file = open(str_json_fn_training, "w")

tmp_json_file.write(json.dumps(tmp_pre_processed_data_set))

tmp_json_file.close()


if os.path.isfile(str_json_fn_testing):
    os.remove(str_json_fn_testing)
else:    ## Show an error ##
    print(" %s file not found", str_json_fn_testing)
    
open(str_json_fn_testing, "w").close()

tmp_json_file = open(str_json_fn_testing, "w")

tmp_json_file.write(json.dumps(tmp_pre_processed_data_set))

tmp_json_file.close()




str_file_path_root = "../../../comp-551-imbd-sentiment-classification/"

str_test_files_path = str_file_path_root + "test/test/"

str_train_files_pos_path = str_file_path_root + "train/train/pos/"

str_train_files_neg_path = str_file_path_root + "train/train/neg/" 



all_test_files = []

train_files_pos = []

train_files_neg = []



logger.info("start enum %s folder. ", str_test_files_path)

if b_partial:
    all_test_files = [f for f in listdir(str_test_files_path)[:i_partial_count] if isfile(join(str_test_files_path, f))]
else:
    all_test_files = [f for f in listdir(str_test_files_path) if isfile(join(str_test_files_path, f))]   

logger.info("end enum %s folder. ", str_test_files_path)



logger.info("start enum %s folder. ", str_train_files_pos_path)

if b_partial:
    train_files_pos = [f for f in listdir(str_train_files_pos_path)[:i_partial_count] if isfile(join(str_train_files_pos_path, f))]
else:
    train_files_pos = [f for f in listdir(str_train_files_pos_path) if isfile(join(str_train_files_pos_path, f))]

logger.info("end enum %s folder. ", str_train_files_pos_path)



logger.info("start enum %s folder. ", str_train_files_neg_path)

if b_partial:
    train_files_neg = [f for f in listdir(str_train_files_neg_path)[:i_partial_count] if isfile(join(str_train_files_neg_path, f))]
else:
    train_files_neg = [f for f in listdir(str_train_files_neg_path) if isfile(join(str_train_files_neg_path, f))]

logger.info("end enum %s folder. ", str_train_files_neg_path)



i_all_test_file_count = len(all_test_files)
print("i_all_test_file_count = ", i_all_test_file_count)

print(all_test_files[10])



i_train_file_pos_count = len(train_files_pos)
print("i_train_file_pos_count = ", i_train_file_pos_count)

print(train_files_pos[10])




i_train_file_neg_count = len(train_files_neg)
print("i_train_file_neg_count = ", i_train_file_neg_count)

print(train_files_neg[10])



X_list_raw = []
Y_list_raw = []

all_files_list_raw = []


logger.info("begin pre-process for pos training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw = pre_process_data_files(str_json_fn_training, str_train_files_pos_path, 1, train_files_pos)





X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for pos training datas... ")




logger.info("begin pre-process for neg training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw = pre_process_data_files(str_json_fn_training, str_train_files_neg_path, 0, train_files_neg)

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


tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw = pre_process_data_files(str_json_fn_testing, str_test_files_path, -1, all_test_files, True, False)

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)
    
logger.info("end pre-process for pos testing datas... ")


