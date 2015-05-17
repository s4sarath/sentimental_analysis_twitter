from flask import Flask, render_template, jsonify
from flask import request 

import datetime
import json
app = Flask(__name__)      
 
@app.route('/')
def home():
  
  return render_template('index.html')









@app.route('/getData',methods=['GET','POST'])
def getDataFromElastic():
	print 'in function'
	print request.data
	values = json.loads(request.data)
	key = values['name']
	# result=start_listener(key)
	# return jsonify(result) 

# if __name__ == '__main__':
#   app.run(port=3125,debug=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)  
