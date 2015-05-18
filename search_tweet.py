


# -*- coding: utf-8 -*-

from twitter_predict import start_predict

from tweepy_listen_new import start_tweet



def search_text():
	'Here we are in serach text'
	import json
	false = False
	true = True

	null = 'null'

	read_file = open('fetched_tweets.txt','r')
	'Here we finished reading'
	text_seg = json.load(read_file)

	try:
		return text_seg['text']

	except:
		raise 'Error no tweets'		



def first_start():
	# start_tweet()

	print 'Done 1'


	text = search_text()

	return start_predict(text)

# key_word = raw_input("enter the key word \n")
# key_word = 'python'
if __name__ == 'main':
	print first_start()	

