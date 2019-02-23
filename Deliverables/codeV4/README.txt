

--------------------------------------------------------------------
comp 551 -- project 2
--------------------------------------------------------------------



Overall, the major packages we use in the project, consist of sklearn related packages, nltk related packages, BeautifulSoup (bs4) package, unicodedata package, numpy, re and math packages. Detailed as following:


data_load_and_preprocessing_adv_stat.py -- the script which does data load and pre-process, support the features such as cleanup the comment, stop word removing, lemmatize, stemming, NWN (next word negate), punctuation removing, separate words from punctuations, filter or tag sentiment words as well as their negation, according to Bing Liu's "opinion lexicon".

imdb_sentiment_svm_linear_svc.py -- the code for using sklearn LinearSVC model. Support feature extracting of BOW (bag of words) and tfidf. Support several options, including cross validatoin by using cross_validate, model selection using GridSearchCV, and sklearn pipeline.

imdb_sentiment_logistic_regression.py -- the code for using sklearn LogisticRegression model. Support feature extracting of BOW (bag of words) and tfidf. Support several options, including cross validatoin by using cross_validate, model selection using GridSearchCV, and sklearn pipeline.


Bernoulli_NB.py -- our implementation of Bernoulli Naive Bayes Model. We only imported "math" and "re" (regex) packages for this implementation.


imdb_sentiment_bernoulli_nb.py -- the script which call our implemented Bernoulli Naive Bayes Model, does not use any sklearn library, except the sklearn metrics and train_test_split, to make sure the input (data set spliting) and the output (performance metrics evaluation) are consistent with the other two models, such that logistic regression and linear svc (SVM) . All the major algorithm of Bernoulli Naive Bayes are implemented from scratch.


positive-words.txt -- the positive words list from Bing Liu's "opinion lexicon".

negative-words.txt -- the negative words list from Bing Liu's "opinion lexicon"

csw.txt -- the list of stop words we use for stop words removing



Instructions to replicate your results

1. run data_load_and_preprocessing_adv_stat.py to load, pre-process the imdb reviews data and generate the intermediate data in .json format, for further fit/predict stage.

2. run imdb_sentiment_svm_linear_svc.py to use the intermediate data and fit/predict with linear svc algorithm, which could generate the best result we have so far, in .csv format,  for the kaggle competition.


 

