from flask import render_template, flash, redirect, request, jsonify,Flask
#from app import app
from preprocessor import Preprocessor as img_prep
import json
import sys
import pickle
import numpy as np 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	app.logger.debug("Went to OCR page")
	return render_template('index.html', title='Optical Character Recognition', prediction=None)


@app.route('/_do_ocr', methods=['GET', 'POST'])
def do_ocr():
	"""Add two numbers server side, ridiculous but well..."""
	app.logger.debug("Accessed _do_ocr page with image data")
	# flash('Just hit the _add_numbers function')
	# a = json.loads(request.args.get('a', 0, type=str))
	data = request.args.get('imgURI', 0, type=str)
	app.logger.debug("Data looks like " + data)
	
	pp = img_prep()
	clf = pickle.load(open('/home/l-ubuntus/Documents/code/html/app_conv_test/finalized_model.sav','rb'))
	char_prediction= clf.predict([pp.preprocess(data)])[0]

	result = "You entered a: %d"%char_prediction

	app.logger.debug("Recognized a character")
	return jsonify(result=result)

if __name__ == '__main__':
	app.run()