# -*- coding: utf-8 -*-
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from collections import Counter
import re
import nltk
# from nltk.corpus import stopwords
import pandas as pd
# import numpy as np

# stopwords = set(stopwords.words("english"))

stopwords = {u'a',
 u'about',
 u'above',
 u'after',
 u'again',
 u'against',
 u'all',
 u'am',
 u'an',
 u'and',
 u'any',
 u'are',
 u'as',
 u'at',
 u'be',
 u'because',
 u'been',
 u'before',
 u'being',
 u'below',
 u'between',
 u'both',
 u'but',
 u'by',
 u'can',
 u'did',
 u'do',
 u'does',
 u'doing',
 u'don',
 u'down',
 u'during',
 u'each',
 u'expect',
 u'few',
 u'for',
 u'from',
 u'further',
 u'had',
 u'has',
 u'have',
 u'having',
 u'he',
 u'her',
 u'here',
 u'hers',
 u'herself',
 u'him',
 u'himself',
 u'his',
 u'how',
 u'i',
 u'if',
 u'in',
 u'into',
 u'is',
 u'it',
 u'its',
 u'itself',
 u'just',
 u'me',
 u'more',
 u'most',
 u'my',
 u'myself',
 u'new',
 u'no',
 u'nor',
 u'not',
 u'now',
 u'of',
 u'off',
 u'on',
 u'once',
 u'only',
 u'or',
 u'other',
 u'our',
 u'ours',
 u'ourselves',
 u'out',
 u'over',
 u'own',
 u's',
 u'same',
 u'she',
 u'should',
 u'since',
 u'so',
 u'some',
 u'such',
 u't',
 u'than',
 u'that',
 u'the',
 u'their',
 u'theirs',
 u'them',
 u'themselves',
 u'then',
 u'there',
 u'these',
 u'they',
 u'this',
 u'those',
 u'through',
 u'to',
 u'too',
 u'under',
 u'until',
 u'up',
 u'very',
 u'was',
 u'we',
 u'were',
 u'what',
 u'when',
 u'where',
 u'which',
 u'while',
 u'who',
 u'whom',
 u'why',
 u'will',
 u'with',
 u'you',
 u'your',
 u'yours',
 u'yourself',
 u'yourselves'}

at_remove = re.compile(ur'@[_\s]{0,2}[a-zA-Z\d_]+\s{1}')
extra_remove = re.compile(ur"[^a-zA-Z]")
def remove_at(sent):
    
    
        
    prep_sent = re.sub(at_remove,'',sent)
    prep_sent = re.sub(extra_remove,' ',prep_sent)
    twitter_preprocessed = prep_sent
    return twitter_preprocessed  


def tokenize(sent):
    
    
        
    tokens = nltk.word_tokenize(sent)
      
    final_tokens = [w.lower() for w in tokens if not w in stopwords and  len(w)>1]
        
    tokenized_twitter = final_tokens
    return tokenized_twitter    


def find_repeating_strings(sent):

	
	multiple_str_count = []

	    
	match = re.match(ur'(\w*([a-z])\2+\w*)',sent,re.IGNORECASE)
	if match:
	    multiple_str_count.append(1)
	else:
	    multiple_str_count.append(0)

	return multiple_str_count        
        
def main_preprocess(text):



	twitter_preprocess = remove_at(text)  
	twitter_tokens = tokenize(twitter_preprocess)
	twitter_token_sent = [' '.join(twitter_tokens)]
	# count_vect = CountVectorizer()
	# train_data_features = count_vect.fit_transform(twitter_token_sent)
	# train_features = train_data_features.toarray()

	# (row,col) = train_features.shape

	# train_features = train_features.reshape(col)

	feature_dict = dict((i,twitter_tokens.count(i)) for i in twitter_tokens)




	vocab_vector = joblib.load('sklearn_model/count_vector_sent.pkl') 
	vocabulary = vocab_vector.vocabulary_
	feature_vector = [0]*len(vocabulary)

	for token in twitter_tokens:

		if token in vocabulary.keys():
			feature_vector[vocabulary[token]] = feature_dict[token]




	multiple_str_count = find_repeating_strings(twitter_preprocess)
	X_test = pd.DataFrame(feature_vector).transpose()

	X_test = pd.concat([X_test,pd.DataFrame(multiple_str_count)],axis=1)
	# print 'X_test', X_test
	# print 'Shape', X_test.shape
	return X_test


def check_sentiment(text):
	text = text.encode('utf-8')
	text = str(text)
	# print 'Text yeshhhhh', text
	# print 'The type', type(text)
	if isinstance(text, str):
		
		return main_preprocess(text)

def start_predict(text):
	test_vector = check_sentiment(text)
	# print 'Test Vector', test_vector
	clf = joblib.load('sklearn_model/random_forest_sent.pkl') 
	result = clf.predict(test_vector)
	# print 'Result', result
	# print 'Lne', len(result)
	# print 'RR', result[0]
	if len(result):
		print 'Yes inside'
		if result[0] == 0:
			return "The text is negative"
		else:
			return "The text is positive"	


if __name__ == 'main':
	pass
	# start_predict('''@SneakerShouts: Sick shot of the Nike Air Python in "Brown Snakeskin"

# Sizes available on Nike here -&gt; http://t.co/neluoxm91s http://t.c…''')