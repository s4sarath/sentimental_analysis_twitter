from flask import Flask, render_template, jsonify
from flask import request 
import os
import time
from tweepy_listen_new import start_tweet
from search_tweet import first_start

import datetime
import json
app = Flask(__name__)      
 

# def search_twwets():



@app.route('/')
def home():
  
  return render_template('index.html')









@app.route('/getData',methods=['GET','POST'])
def search_twwets():
	print 'called'
	# print 'in function'
	# print request.data
	polarity = ''
	values = json.loads(request.data)
	print 'values',values
	search_text = values['search_text']
	if search_text:
		print search_text
		res = start_tweet(search_text)
		print 'Res', res
		time.sleep(10)
		if res:
			print 'under res'
			polarity = first_start()
			print polarity

		# result=start_listener(key)
		return polarity

# if __name__ == '__main__':
#   app.run(port=3125,debug=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)  
