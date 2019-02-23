# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:00:29 2019

@author: PeterXu
"""

import math
import re

def extract_features(in_str_text):

    #out_str_text = re.sub(r'[^a-zA-Z0-9]', " ", in_str_text)    
    out_str_text_lc = in_str_text.lower()
    
    out_feature_list = out_str_text_lc.split()
    
    return out_feature_list

class Bernoulli_NaiveBayes_Classifier(object):

    def __init__(self):
        self.log_priors = None
        self.cond_probs = None
        self.features = None


    def fit(self, X_texts, Y_classes):

        class_stat = {}        
     
        for class_item in Y_classes:
            if class_item not in class_stat:
                class_stat[class_item] = 1
            else:
                class_stat[class_item] += 1
            
        
        print("\n\n class_stat =", class_stat)
        
        N = float(sum(class_stat.values()))
        
        self.log_priors = {key: math.log(val / N) for key, val in class_stat.items()}
        
        print("\n\n self.log_priors = ", self.log_priors)


        
        X_features_set = []
        
        
        i_index = 0
        
        
        print("\n start feature extraction")
        
        for text_item in X_texts:
            
            feature_list = extract_features(text_item)
            
            feature_set = set(feature_list)
            
            X_features_set.append(feature_set)
            
            i_index += 1
            
            
        
        print("\n begin summerize features ")

        self.features = set([])
        
        for features_item in X_features_set:
            
            self.features = self.features | set([feature_item for feature_item in features_item])
            
       
        self.cond_probs = {}
                
        for class_item in self.log_priors:
            
            self.cond_probs[class_item] = {}
            
            for feature_item in self.features:
                self.cond_probs[class_item][feature_item] = 0.0
            
        
        print("\n\n self.features = ", self.features)
        
        
        print("\n\n len of self.features = ", len(self.features))
        
        print("\n initialized cond_probs")

        
        i_index = 0
        
        for i_index in range(0, len(X_features_set), 1):
            
            class_item = Y_classes[i_index]
            features_item = X_features_set[i_index]
        
            for feature_item in features_item:
                self.cond_probs[class_item][feature_item] += 1.0
                
            i_index += 1
            
            if i_index % 1000 == 0:
                print("\n i_index = ", i_index)              

        
        print("\n counted cond_probs")

        # caluculate conditional probability with laplace smoothing:
        for class_item in self.cond_probs:
            N = class_stat[class_item]
            
            for feature_item, val in self.cond_probs[class_item].items():
                
                self.cond_probs[class_item][feature_item] = (val + 1.0) / (N + 2.0)

        
        print("\n calculated cond_probs")
        
       
    def predict_one_item(self, text):

        feature_list = extract_features(text)
        
        feature_set = set(feature_list)

        pred_class = None
        max_ = float("-inf")
        
        

        for class_item in self.log_priors:
            log_sum = self.log_priors[class_item]
            
            for feature_item in self.features:
                prob = self.cond_probs[class_item][feature_item]
                
                
                tmp_prob = 0.0
                if feature_item in feature_set:
                    tmp_prob = prob
                else:
                    tmp_prob = 1.0 - prob                
                    
                log_sum += math.log(tmp_prob)
                
            if log_sum > max_:
                max_ = log_sum
                pred_class = class_item

        return pred_class
    
    
    
    def predict(self, X_texts):       
        
        out_Y_pred_classes = []
        
        i_index = 0
        
        for text_item in X_texts:
            
            out_Y_pred_classes.append(self.predict_one_item(text_item))
            
            i_index += 1
            
            if i_index % 500 == 0:
                print("\n predicted so far, i_index = ", i_index)
        
        return out_Y_pred_classes
        

    