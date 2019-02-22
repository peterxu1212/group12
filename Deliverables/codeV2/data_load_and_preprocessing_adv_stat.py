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

from nltk.tokenize import sent_tokenize


from nltk.corpus import stopwords

#from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer

import unicodedata


def remove_stop_words(words, b_conservative=False):
    
    filtered_words = []
    if conservative_sw:
        filtered_words = [word for word in words if word not in conservative_sw]
    else:
        filtered_words = [word for word in words if word not in nltk_eng_stop_words_revised]
    
    return filtered_words

def cleanpunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]', r' ', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return  cleaned


def get_detail_context(str_error, str_text, i_context_range=10):
      
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
    
    out_str_text = in_str_text
    if not b_cleanup:
        return out_str_text
    
    #'
    out_str_text = re.sub(r'[\u2019]', '\'', out_str_text, re.UNICODE)
    out_str_text = re.sub(r'[\u2018]', '\'', out_str_text, re.UNICODE)
    
    #-
    out_str_text = re.sub(r'[\u2013]', '', out_str_text, re.UNICODE)
    
    #"
    out_str_text = re.sub(r'[\u201c]', '', out_str_text, re.UNICODE)
    #"
    out_str_text = re.sub(r'[\u201d]', '', out_str_text, re.UNICODE)
    
    
    #file_content_wo_html_tags = re.sub(r'[\u0308]', '', file_content_wo_html_tags, re.UNICODE)
    
    
    out_str_text = re.sub(r'[\u0131]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\u3001]', '', out_str_text, re.UNICODE)
    
    
    out_str_text = re.sub(r'[\x80-\xfe]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\x96]', '', out_str_text, re.UNICODE)
    out_str_text = re.sub(r'[\x9e]', '', out_str_text, re.UNICODE)
    out_str_text = re.sub(r'[\xab]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\x85]', '', out_str_text, re.UNICODE)                
    
    out_str_text = re.sub(r'[\x8e]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\xbb]', '', out_str_text, re.UNICODE)
    
    
    out_str_text = re.sub(r'[\x80-\xfe]', '', out_str_text, re.UNICODE)
    
    
    out_str_text = re.sub(r'[\u20a4]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\u2227]', '', out_str_text, re.UNICODE)
    
    out_str_text = re.sub(r'[\u25bc]', '', out_str_text, re.UNICODE)
    
    
    
    out_str_text = re.sub(r'[\uf04a]', '', out_str_text, re.UNICODE)
    
    
    out_str_text = re.sub(r'[\uf0b7]', '', out_str_text, re.UNICODE)
    
    
    
    return out_str_text




def normalize_sentiment(in_str_item):
    
    out_str_item = ""
           
    if in_str_item in pos_words_filter:
        #out_str_item = " imdbpos " + in_str_item + " imdbpos "
        out_str_item = " imdbpos "
    
    if in_str_item in neg_words_filter:
        #out_str_item = " imdbneg " + in_str_item + " imdbneg "
        out_str_item = " imdbneg "
        
    
    if out_str_item == "":
        out_str_item = in_str_item
    
    return out_str_item



def count_specific_term(in_text):
    
    i_pos_cnt = 0
    i_neg_cnt = 0
    
    tnd_words = word_tokenize(in_text)
    
    for word in tnd_words:
        if word == "good" or word == "great" or word == "imdbpos":
            i_pos_cnt += 1
        elif word == "bad" or word == "imdbneg":
            i_neg_cnt += 1
        else:
            pass
      
    i_raw_overall_sentiment = i_pos_cnt - i_neg_cnt
    
    i_overall_sentiment_calc = 0
    
    if i_raw_overall_sentiment > 0:
        i_overall_sentiment_calc = 1
    elif i_raw_overall_sentiment < 0:
        i_overall_sentiment_calc = -1
    else:
        i_overall_sentiment_calc = 0
        
    return i_overall_sentiment_calc
    


def calc_overall_sentiment(in_sentiment_calc_text):
    
    tnd_sentences = sent_tokenize(in_sentiment_calc_text)
    
    #(';', ':', ',', '.', '!', '?')   
    i_overall_sentiment = 0

    i_index = 0
    for sentence_item in tnd_sentences:
   
        sub_sentence_items = re.split(r'([.;:,!?])', sentence_item)
        
        #print("\n sub_sentence_items = ", sub_sentence_items)
        for sub_sentence_item in sub_sentence_items:
            i_overall_sentiment += count_specific_term(sub_sentence_item)
        
        i_index += 1
        
        
    
    return i_overall_sentiment


def further_process_data(in_str_text):

   
    
    out_str_text = in_str_text
    
    
    #re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',s)
    
    out_str_emoticons = "none"
    list_ret = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', out_str_text, re.UNICODE)
    
    if len(list_ret) > 0:        
        
        out_str_emoticons = str(list_ret)
        
    """    
        for item in list_ret:
            if "(" in item:
                #print("\n", item)
        
                out_str_emoticons = "imdbemoticonfrown"
                break
                #str(list_ret)
    """
    
    if b_separate_wp:
        #extend space for punctuations
        out_str_text = re.sub(r"\.", " . ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\,", " , ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\!", " ! ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\?", " ? ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\:", " : ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\;", " ; ", out_str_text, re.UNICODE)
        out_str_text = re.sub(r"\"", " \" ", out_str_text, re.UNICODE)
        #out_str_text = re.sub(r"\'", " \' ", out_str_text, re.UNICODE)
        
        
    tmp_sentiment_calc_text = out_str_text
    
    if b_separate_wp:
        out_str_text = re.sub(r"\'", " \' ", out_str_text, re.UNICODE)
        
    if b_wo_sw:
        raw_word_list = tmp_sentiment_calc_text.split()
    
        raw_word_list = remove_stop_words(raw_word_list, True)
    
        tmp_sentiment_calc_text = " ".join(raw_word_list)
    
        
        raw_word_list = out_str_text.split()
        
        raw_word_list = remove_stop_words(raw_word_list, True)
    
        out_str_text = " ".join(raw_word_list)

    
    str_nwn = ""
    
    if b_negate:

        #pass
        tmp_sentiment_calc_text = re.sub(r"n\'t ", " not ", tmp_sentiment_calc_text, re.UNICODE)
        tmp_sentiment_calc_text = re.sub(r" never ", " not ", tmp_sentiment_calc_text, re.UNICODE)
        tmp_sentiment_calc_text = re.sub(r" no ", " not ", tmp_sentiment_calc_text, re.UNICODE)
        #tmp_sentiment_calc_text = re.sub(r" nothing ", " not ", tmp_sentiment_calc_text, re.UNICODE)
        
        
        
        tmp_sentiment_calc_text = re.sub(r"\'", " \' ", tmp_sentiment_calc_text, re.UNICODE)
        
        tmp_sentiment_calc_text = " ".join(tmp_sentiment_calc_text.split())
        

        
        
        if b_w_nwn:
            str_nwn = tmp_sentiment_calc_text
        
    
    if b_w_nwn:
        
        #remove single character
        
        str_nwn =  re.sub(r"\b[a-zA-Z]\b", "", str_nwn, re.UNICODE)
        
        wl_nwn = str_nwn.split()
        
        new_wl_nwn = []
        
        tmp_word = ""
        for word in wl_nwn:
            
            if tmp_word == "not":
                word = tmp_word + word
                tmp_word = ""
                new_wl_nwn.append(word)
                
            else:
                   
                if word == "not":
                    tmp_word = word
                else:
                    new_wl_nwn.append(word)
                
            str_nwn = " ".join(new_wl_nwn)
            
    
    
    i_overall_sentiment = 0
    
    if b_sentiment_words_filter:
        #pass
        tnd_words = word_tokenize(tmp_sentiment_calc_text)
        normalized_tnd_words = map(normalize_sentiment, tnd_words)
        
        tmp_sentiment_calc_text = " ".join(normalized_tnd_words)
        
        tmp_sentiment_calc_text = re.sub(r" not imdbpos ", " imdbneg ", tmp_sentiment_calc_text, re.UNICODE)        
        tmp_sentiment_calc_text = re.sub(r" not imdbneg ", " imdbpos ", tmp_sentiment_calc_text, re.UNICODE)

        
        i_overall_sentiment = calc_overall_sentiment(tmp_sentiment_calc_text)
        


    
    if b_wo_punctuation:
        #out_str_text = re.sub(r'[^\w\s]', ' ', out_str_text, re.UNICODE)
        out_str_text = re.sub(r'[^\w\s]', ' ', out_str_text)
        tmp_sentiment_calc_text = re.sub(r'[^\w\s]', ' ', tmp_sentiment_calc_text)
        
        str_nwn = re.sub(r'[^\w\s]', ' ', str_nwn)
    
    
    #print("\n\n\n\n before lemmatize: ", out_str_text)
    
    
    tnd_words = word_tokenize(out_str_text)
    
    
    out_str_nums = "none"
    
    filtered_nums = [word for word in tnd_words if word in specific_numbers]
    
    if len(filtered_nums) > 0:
        #if "10" in filtered_nums:
        out_str_nums = " ".join(filtered_nums)
    
    
    senti_tnd_words = word_tokenize(tmp_sentiment_calc_text)
    
    
    nwn_tnd_words = word_tokenize(str_nwn) 
    
    #
    #stemmer = PorterStemmer()
 
    #tnd_words = [stemmer.stem(word) for word in tnd_words]
    
    if b_w_stemming:
        
        stemmer = SnowballStemmer('english')
        
        tnd_words = [stemmer.stem(word) for word in tnd_words]
        
        senti_tnd_words = [stemmer.stem(word) for word in senti_tnd_words]
        
        nwn_tnd_words = [stemmer.stem(word) for word in nwn_tnd_words]
        
        
    if b_w_lemmatize:
    
        lemmatizer = WordNetLemmatizer() 
        
        tnd_words = [lemmatizer.lemmatize(word) for word in tnd_words]
        
        senti_tnd_words = [lemmatizer.lemmatize(word) for word in senti_tnd_words]
    
        nwn_tnd_words = [lemmatizer.lemmatize(word) for word in nwn_tnd_words]
    
    
    
        
    out_str_text = " ".join(tnd_words)
    
    tmp_sentiment_calc_text = " ".join(senti_tnd_words)
    
    str_nwn = " ".join(nwn_tnd_words)
    
    #out_str_text = str_text_wo_punctuation

    return out_str_text, i_overall_sentiment, tmp_sentiment_calc_text, out_str_nums, out_str_emoticons, str_nwn



def review_to_wordlist(review_text, remove_stopwords=False):

    #tmp_soup = BeautifulSoup(review, 'html.parser')
    
    # Get text only
    #review_text = tmp_soup.get_text()
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                                   
    # searching in a set rather than a list is faster in python
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def pre_process_data_files(in_str_json_fn, in_str_file_path, in_int_y_val, in_file_name_list = [], b_guess_when_exception=False, b_test=False, b_clean_up=False):
    
    pre_processed_data_set = []
    
    out_pred_list_raw = []
    
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
        str_senti_text = ""
        
        tmp_processed_text = ""
        
        pre_processed_item = {}
        
        error_msg_item = {}
        
        i_oa_sentiment = 0
        
        str_nums = "none"
        
        str_emoticons = "none"
        
        str_nwn = ""
    
        try:
                        
            if not b_test:
                
                #file_handle = open(str_full_file_name, encoding = 'gbk', errors = 'ignore')
                #file_handle = open(str_full_file_name, encoding = 'gb18030', errors = 'ignore')
                file_handle = open(str_full_file_name, encoding = 'utf-8')
                
                statinfo = os.stat(str_full_file_name)
        
        
                pre_processed_item['file_size'] = statinfo.st_size

                file_content = file_handle.read()
                
                pre_processed_item['raw_text'] = file_content
                
                
                
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n file_content = ", file_content)
                
                
                tmp_soup = BeautifulSoup(file_content, 'html.parser')
                
                file_content_wo_html_tags = tmp_soup.get_text()
                
                
                
                pre_processed_item['text_simple_cleanup'] = " ".join(review_to_wordlist(file_content_wo_html_tags, remove_stopwords=True))
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n file_content_wo_html_tags = ", file_content_wo_html_tags)
                
                
                file_content_wo_html_tags_lc = file_content_wo_html_tags.lower()
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n file_content_wo_html_tags_lc = ", file_content_wo_html_tags_lc)
                
                
                #str_text_cleaned = clean_up_data(file_content_wo_html_tags_lc)
                
                str_text_cleaned = file_content_wo_html_tags_lc
                
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n str_text_cleaned = ", str_text_cleaned)
                
                
                tmp_processed_text = str_text_cleaned
                
                
                #file_content_wo_html_tags = re.sub(r'[\x84]', '', file_content_wo_html_tags, re.UNICODE)
                #file_content_wo_html_tags = re.sub(r'[\x85]', '', file_content_wo_html_tags, re.UNICODE)
                
                #\u0400-\u0500
                
                #file_content_wo_html_tags = file_content_wo_html_tags.replace("naïve", "naive")
                
                
                
                
                
                
                code_converted = unicodedata.normalize('NFKD', str_text_cleaned).encode('ascii', errors='ignore')
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n code_converted = ", code_converted)
                
                uni_code_converted = code_converted
                if isinstance(code_converted, bytes):
                    uni_code_converted = str(code_converted, encoding='utf-8');
                
                
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n uni_code_converted = ", uni_code_converted)
                
                #str_processed_text = uni_code_converted
                
                str_processed_text, i_oa_sentiment, str_senti_text, str_nums, str_emoticons, str_nwn = further_process_data(uni_code_converted)
                
                
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n str_processed_text = ", str_processed_text) 
                    print("\n\n\n\n\n\n str_senti_text = ", str_senti_text)
                    print("\n\n\n\n\n\n str_nwn = ",  str_nwn)
                
                            
                
            
                
            else:
                str_processed_text = "bad"
            
        except Exception as e:
            
            str_error = str(e)
            
            if type(e) is UnicodeEncodeError:
            
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
            
            else:
                
                logger.warning("other error !!!!!! %s ", str_error)
                logger.warning(type(e))
                
            #print(str_tmp)
            
            if not b_guess_when_exception:
                continue
            else:
                #str_processed_text = "bad"
                
                
                
                code_converted = unicodedata.normalize('NFKD', tmp_processed_text).encode('ascii', errors='ignore')
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n exception str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n code_converted = ", code_converted)
                
                uni_code_converted = code_converted
                if isinstance(code_converted, bytes):
                    uni_code_converted = str(code_converted, encoding='utf-8');
                            
                    
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n exception str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n uni_code_converted = ", uni_code_converted)
                
                #str_processed_text = uni_code_converted
                
                str_processed_text, i_oa_sentiment, str_senti_text, str_nums, str_emoticons, str_nwn = further_process_data(uni_code_converted)
                
                
                if str_file_name == "10327_7.txt":
                    print("\n\n\n\n\n\n exception str_file_name = ", str_file_name)
                    print("\n\n\n\n\n\n str_processed_text = ", str_processed_text)
                    print("\n\n\n\n\n\n str_senti_text = ", str_senti_text)
                    print("\n\n\n\n\n\n str_nwn = ",  str_nwn)
                
                #str_processed_text = tmp_processed_text
                
        out_pred_list_raw.append(i_oa_sentiment)
           
        out_X_list_raw.append(str_processed_text)
        out_Y_list_raw.append(in_int_y_val)
        out_files_list_raw.append(str_full_file_name)
        out_pure_files_list_raw.append(str_file_name)
        
        pre_processed_item['text'] = str_processed_text
        pre_processed_item['senti_text'] = str_senti_text
        pre_processed_item['str_nwn'] = str_nwn
        
        
        pre_processed_item['y_val'] = in_int_y_val
        pre_processed_item['full_file_name'] = str_full_file_name
        
        pre_processed_item['i_sentiment_estimate'] = i_oa_sentiment
        
        pre_processed_item['str_nums'] = str_nums
        pre_processed_item['str_emoticons'] = str_emoticons
                
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
    
       
    
    
    
    return out_X_list_raw, out_Y_list_raw, out_files_list_raw, out_pure_files_list_raw, str_output_exceptions, out_pred_list_raw









str_ec = "encode character \'\\"
str_in_position = "\' in position "







fout_t2w = open('./data_preprocess_results.txt', 'w', buffering=1)



i_partial_count = 500

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


#b_negate = False

#separate words and punctuations
b_separate_wp = True




pos_word_file_name = "positive-words"
neg_word_file_name = "negative-words"

pos_words_filter = {}
        
wc_index = 0

str_sentiment_file_name = '../' + pos_word_file_name + '.txt'
with open(str_sentiment_file_name,'r') as rf_words:
    
    while True:
        # read line
        line = rf_words.readline()
        #print(line)

        # check if line is not empty
        if not line:
            break
        
        lhs = line.split()        
        
        pos_words_filter[lhs[0]] = 1
        #wc_index += 1
        #print(lhs, rhs)
        
    rf_words.close()
    



neg_words_filter = {}
        
wc_index = 0

str_sentiment_file_name = '../' + neg_word_file_name + '.txt'
with open(str_sentiment_file_name,'r') as rf_words:
    
    while True:
        # read line
        line = rf_words.readline()
        #print(line)

        # check if line is not empty
        if not line:
            break
        
        lhs = line.split()        
        
        neg_words_filter[lhs[0]] = 1
        #wc_index += 1
        #print(lhs, rhs)
        
    rf_words.close()
    




conservative_sw = {}
        
wc_index = 0

str_csw_file_name = '../' + "csw" + '.txt'
with open(str_csw_file_name,'r') as rf_words:
    
    while True:
        # read line
        line = rf_words.readline()
        #print(line)

        # check if line is not empty
        if not line:
            break
        
        lhs = line.split()        
        
        conservative_sw[lhs[0]] = 1
        #wc_index += 1
        #print(lhs, rhs)
        
    rf_words.close()
    

print("\n\n conservative_sw = ", conservative_sw)

print("\n\n pos_words_filter = ", pos_words_filter)

print("\n\n neg_words_filter = ", neg_words_filter)






logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")



nltk_eng_stop_words = set(stopwords.words('english'))

print("nltk_eng_stop_words = \n", nltk_eng_stop_words)









#nltk_eng_stop_words_revised = {'she', 'and', 'he', "you'd", 'itself', 'as', 'being', 'with', 'are', 'ours', "you're", 'd', 'did', 'it', 'hers', 'have', 'her', "she's", 'doing', 'be', 'because', "you'll", 'me', 'to', 'while', 'the', 'such', 'whom', 'yourselves', 'a', 'its', 'am', 'him', 'his', 'them', 'why', 'how', 'yourself', 'if', 'in', 'into', 'myself', 'do', "you've", 'had', 'when', 'some', 'yours', 'who', 'been', 'himself', 'where', 'has', 'these', 've', 'ourselves', 'your', "that'll", 'those', 'at', 'then', 'was', 'i', 'theirs', 'is', 'they', 'there', 'does', 'through', 'so', 'having', 'that', 'our', 'their', 'or', 'themselves', 's', 'about', 'an', 'which', 'we', 'on', 'my', 'here', 'were', 'that', 'this', 'you', 'herself', 'from', "it's", 'for', 'between', 'by', 'of'}

nltk_eng_stop_words_revised = {'she', 'and', 'he', "you", 'itself', 'as', 'with', 'are', 'ours', "you're", 'did', 'it', 'hers', 'her', 'doing', 'be', 'because', "you'll", 'me', 'to', 'while', 'the', 'such', 'whom', 'yourselves', 'a', 'its', 'am', 'him', 'his', 'them', 'why', 'how', 'yourself', 'if', 'in', 'into', 'myself', 'do', "you've", 'had', 'when', 'some', 'yours', 'who', 'been', 'himself', 'where', 'has', 'these', 've', 'ourselves', 'your', "that'll", 'those', 'at', 'then', 'was', 'i', 'theirs', 'is', 'they', 'there', 'does', 'through', 'so', 'having', 'that', 'our', 'their', 'or', 'themselves', 'about', 'an', 'which', 'we', 'on', 'my', 'here', 'were', 'that', 'this', 'you', 'herself', 'from', "it's", 'for', 'between', 'by', 'of'}


specific_numbers = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"} 

print("nltk_eng_stop_words_revised = \n", nltk_eng_stop_words_revised)

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
    

str_fn_postfix += "_simplified"


str_json_fn_training = "../" + "training" + "_set" + str_fn_postfix + ".json"

str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"



logger.info("\n re-generate json files %s and %s \n\n", str_json_fn_training, str_json_fn_testing)


tmp_pre_processed_data_set = []

tmp_pre_processed_item = {}

tmp_pre_processed_item['text'] = ""
tmp_pre_processed_item['y_val'] = -2
tmp_pre_processed_item['full_file_name'] = ""
tmp_pre_processed_item['id'] = ""

tmp_pre_processed_item['i_sentiment_estimate'] = 0
        
tmp_pre_processed_data_set.append(tmp_pre_processed_item)



tmp_err_set = []

tmp_err_item = {}


tmp_err_item['error_msg'] = ""
tmp_err_item['error_unicode_num'] = -2
tmp_err_item['error_unicode_str'] = ""


tmp_err_set.append(tmp_err_item)


str_err_fn = "./process_err_msg_set" + str_fn_postfix + ".json"


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



pred_list_raw = []
X_list_raw = []
Y_list_raw = []

all_files_list_raw = []


logger.info("begin pre-process for pos training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw, tmp_pred_list_raw  = pre_process_data_files(str_json_fn_training, str_train_files_pos_path, 1, train_files_pos, True)



#fout_t2w.write(str_output_tw)
pred_list_raw.extend(tmp_pred_list_raw)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for pos training datas... ")




logger.info("begin pre-process for neg training datas... ")

tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw, tmp_pred_list_raw = pre_process_data_files(str_json_fn_training, str_train_files_neg_path, 0, train_files_neg, True)


#fout_t2w.write(str_output_tw)
pred_list_raw.extend(tmp_pred_list_raw)

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

all_files_list_raw.extend(tmp_files_list_raw)
    
logger.info("end pre-process for neg training datas... ")
    


print("len pred_list_raw = \n", len(pred_list_raw))

print("len Y_list_raw = \n", len(Y_list_raw))


    
#print("X_list_raw = \n", X_list_raw)

#print("Y_list_raw = \n", Y_list_raw)



pred_list_real_test_raw = []

X_list_real_test_raw = []
Y_list_real_test_raw = []

all_files_list_real_test_raw = []
all_pure_files_list_real_test_raw = []

logger.info("begin pre-process for testing datas... ")


tmp_list_X_raw, tmp_list_Y_raw, tmp_files_list_raw, tmp_pure_files_list_raw, str_output_tw, tmp_pred_list_raw = pre_process_data_files(str_json_fn_testing, str_test_files_path, -1, all_test_files, True, False)


#fout_t2w.write(str_output_tw)

pred_list_real_test_raw.extend(tmp_pred_list_raw)

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

all_files_list_real_test_raw.extend(tmp_files_list_raw)
all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)








    
    
logger.info("end pre-process for testing datas... ")




