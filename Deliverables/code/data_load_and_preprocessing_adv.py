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

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


import unicodedata


def remove_stop_words(words):
    
    filtered_words = [word for word in words if word not in nltk_eng_stop_words]
    
    return filtered_words




def get_detail_context(str_error, str_text, i_context_range=10):
    
    """ 
    
    wlist = re.findall(r'\d+', str_error)
    
    #print("\n\n wlist = ", wlist)
    
    i_uni_number = int(wlist[0], 16)
    
    #print("\n\n i_uni_number = ", i_uni_number)
    
    i_pos_num = int(wlist[1])
    
    #print("\n\n i_pos_num = ", i_pos_num)
    
    str_chr = chr(i_uni_number)
    
    #print("\n\n str_chr = ", str_chr)
    
    """    
    
    try:

        i_index_uni_num_start = str_error.index(str_ec)       
    
        i_index_pos_start = str_error.index(str_in_position)
    
    except Exception as e:
        
        print(e, type(e),  "\n")
        return "", -1, ""
        
    
    i_index_pos_end = str_error.index(": ordinal not")
    
    
    i_uni_real_start = i_index_uni_num_start + len(str_ec)
    
    str_uni_num = str_error[i_uni_real_start : i_index_pos_start]
    str_pos_num = str_error[i_index_pos_start + len(str_in_position) : i_index_pos_end]
    

    
    i_uni_number = int(str_uni_num[1:], 16)
    
    str_chr = chr(i_uni_number)
    
    
    """
    print("\n\n str_uni_num = ", str_uni_num)

    print("\n\n i_uni_number = ", i_uni_number)
    
    print("\n\n str_pos_num = ", str_pos_num)

    print("\n\n str_chr = ", str_chr)
    """
    
    
    i_pos_num = int(str_pos_num)
    
    i_context_start = 0
    i_context_start = i_pos_num - i_context_range
    
    
    
    if i_context_start < 0:
        i_context_start = 0
    
    i_context_end = 0    
    i_context_end = i_pos_num + i_context_range
    
    if i_context_end > len(str_text):
        i_context_end = len(str_text)
        
    
    
    str_context = str_text[i_context_start : i_context_end]
    
    str_context += str_uni_num + ", str_chr = " + str_chr + " \n"
    
        
    #print("\n\n context: ", )
    
    return str_context, i_uni_number, str_uni_num
    
    

def clean_up_data(in_str_text):
    
    #'
    in_str_text = re.sub(r'[\u2019]', '\'', in_str_text, re.UNICODE)
    in_str_text = re.sub(r'[\u2018]', '\'', in_str_text, re.UNICODE)
    
    #-
    in_str_text = re.sub(r'[\u2013]', '', in_str_text, re.UNICODE)
    
    #"
    in_str_text = re.sub(r'[\u201c]', '', in_str_text, re.UNICODE)
    #"
    in_str_text = re.sub(r'[\u201d]', '', in_str_text, re.UNICODE)
    
    
    #file_content_wo_html_tags = re.sub(r'[\u0308]', '', file_content_wo_html_tags, re.UNICODE)
    
    
    in_str_text = re.sub(r'[\u0131]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\u3001]', '', in_str_text, re.UNICODE)
    
    
    in_str_text = re.sub(r'[\x80-\xfe]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\x96]', '', in_str_text, re.UNICODE)
    in_str_text = re.sub(r'[\x9e]', '', in_str_text, re.UNICODE)
    in_str_text = re.sub(r'[\xab]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\x85]', '', in_str_text, re.UNICODE)                
    
    in_str_text = re.sub(r'[\x8e]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\xbb]', '', in_str_text, re.UNICODE)
    
    
    in_str_text = re.sub(r'[\x80-\xfe]', '', in_str_text, re.UNICODE)
    
    
    in_str_text = re.sub(r'[\u20a4]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\u2227]', '', in_str_text, re.UNICODE)
    
    in_str_text = re.sub(r'[\u25bc]', '', in_str_text, re.UNICODE)
    
    
    
    in_str_text = re.sub(r'[\uf04a]', '', in_str_text, re.UNICODE)
    
    
    in_str_text = re.sub(r'[\uf0b7]', '', in_str_text, re.UNICODE)
    
    out_str_text = in_str_text
    
    return out_str_text
    


def pre_process_data_files(in_str_json_fn, in_str_file_path, in_int_y_val, in_file_name_list = [], b_guess_when_exception=False, b_test=False, b_clean_up=False):
    
    pre_processed_data_set = []
    
    out_X_list_raw = []
    out_Y_list_raw = []
    
    out_files_list_raw = []
    out_pure_files_list_raw = []
    
    out_error_msg_set = []
    
    str_output_exceptions = ""
    
    i_cur_processing_index = 0
        
    #for str_file_name in in_file_name_list[:100]:
    for str_file_name in in_file_name_list:
    
        i_cur_processing_index += 1
        
        if i_cur_processing_index % 500 == 0:
            logger.info("pre_process_data_files i_cur_processing_index = %d", i_cur_processing_index)
        

        str_full_file_name = in_str_file_path + str_file_name
        file_content = ""
        
        str_processed_text = ""
        
        tmp_processed_text = ""
        
        pre_processed_item = {}
        
        error_msg_item = {}
    
        try:
                        
            if not b_test:
                
                #file_handle = open(str_full_file_name, encoding = 'gbk', errors = 'ignore')
                #file_handle = open(str_full_file_name, encoding = 'gb18030', errors = 'ignore')
                file_handle = open(str_full_file_name, encoding = 'utf-8')

                file_content = file_handle.read()
                
                tmp_soup = BeautifulSoup(file_content, 'html.parser')
                
                file_content_wo_html_tags = tmp_soup.get_text()
                
                
                file_content_wo_html_tags_lc = file_content_wo_html_tags.lower()
                
                str_text_cleaned = clean_up_data(file_content_wo_html_tags_lc)
                
                tmp_processed_text = str_text_cleaned
                
                
                #file_content_wo_html_tags = re.sub(r'[\x84]', '', file_content_wo_html_tags, re.UNICODE)
                #file_content_wo_html_tags = re.sub(r'[\x85]', '', file_content_wo_html_tags, re.UNICODE)
                
                #\u0400-\u0500
                
                #file_content_wo_html_tags = file_content_wo_html_tags.replace("naïve", "naive")
                
                
                
                
                
                
                code_converted = unicodedata.normalize('NFKD', str_text_cleaned).encode('ascii')
                
                uni_code_converted = code_converted
                if isinstance(code_converted, bytes):
                    uni_code_converted = str(code_converted, encoding='utf-8');
                
                
                
                
                str_processed_text = uni_code_converted
                
                
                
                
                #word_list = uni_code_converted.split()
                
                #word_list = remove_stop_words(word_list)
                
                #file_content_w_1ws = ' '.join(word_list)
                
                #file_content_pure_english = re.sub("[^a-zA-Z]", " ", file_content_w_1ws)
                
                
                
                #word_list_pure_english = file_content_pure_english.split()
                
                #file_content_pure_english_w_1ws = ' '.join(word_list_pure_english)
                
                #if b_wo_punctuation:
                    #file_content = re.sub(r'[^\w\s]', ' ', file_content)
                    
                str_processed_text = uni_code_converted
                
                
                """
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n str_processed_text = ", str_processed_text)
                
                """
            
                
            else:
                str_processed_text = "bad"
            
        except Exception as e:
            
            str_error = str(e)
            
            str_tmp = "pre_process_data_files Exception e = " + str(type(e)) + ":" + str_error + " when execute encode  for str_full_file_name = " + str(str_full_file_name) + "\n"
        
            str_exception_msg = str_tmp
            
            str_out_of_gdc, i_unicode_num, str_unicode = get_detail_context(str_error, file_content_wo_html_tags)
            
            str_exception_msg += str_out_of_gdc
            
            error_msg_item['error_msg'] = str_exception_msg
            error_msg_item['error_unicode_num'] = i_unicode_num
            error_msg_item['error_unicode_str'] = str_unicode
            
            str_output_exceptions += str_exception_msg
        
            #logger.warning("%s", str_exception_msg)
            
            out_error_msg_set.append(error_msg_item)
            
            #print(str_tmp)
            
            if not b_guess_when_exception:
                continue
            else:
                #str_processed_text = "bad"
                str_processed_text = tmp_processed_text
                
        
           
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
    
    
    
    
    str_json_fn = str_err_fn
    
    jsonFile = open(str_json_fn, "r") # Open the JSON file for reading
    json_data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    logger.info("pre_process_data_files len of json_data = %d, for json file %s ", len(json_data), str_json_fn)

    ## Working with buffered content
    json_data.extend(out_error_msg_set)
    
    logger.info("pre_process_data_files len of json_data = %d, after extend ", len(json_data))


    ## Save our changes to JSON file
    jsonFile = open(str_json_fn, "w")
    jsonFile.write(json.dumps(json_data))
    jsonFile.close()
    
       
    
    
    
    return out_X_list_raw, out_Y_list_raw, out_files_list_raw, out_pure_files_list_raw, str_output_exceptions






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





str_ec = "encode character \'\\"
str_in_position = "\' in position "







fout_t2w = open('./data_preprocess_results.txt', 'w', buffering=1)


#b_partial = True
    
b_partial = False

b_wo_punctuation = True


i_partial_count = 500






logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")



nltk_eng_stop_words = set(stopwords.words('english'))

print("nltk_eng_stop_words = \n", nltk_eng_stop_words)





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



tmp_err_set = []

tmp_err_item = {}


tmp_err_item['error_msg'] = ""
tmp_err_item['error_unicode_num'] = -2
tmp_err_item['error_unicode_str'] = ""


tmp_err_set.append(tmp_err_item)


str_err_fn = "./process_err_msg_set.json"


if os.path.isfile(str_err_fn):
    os.remove(str_err_fn)
else:    ## Show an error ##
    print(" %s file not found", str_err_fn)

    
tmp_json_file = open(str_err_fn, "w")

tmp_json_file.write(json.dumps(tmp_err_set))

tmp_json_file.close()




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

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw = pre_process_data_files(str_json_fn_training, str_train_files_pos_path, 1, train_files_pos, True)



#fout_t2w.write(str_output_tw)


X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for pos training datas... ")




logger.info("begin pre-process for neg training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw = pre_process_data_files(str_json_fn_training, str_train_files_neg_path, 0, train_files_neg, True)


#fout_t2w.write(str_output_tw)

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


tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw = pre_process_data_files(str_json_fn_testing, str_test_files_path, -1, all_test_files, True, False)


#fout_t2w.write(str_output_tw)

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)
    
logger.info("end pre-process for pos testing datas... ")


