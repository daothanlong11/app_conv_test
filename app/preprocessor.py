from __future__ import print_function
import pickle
import numpy as np
import os.path
import base64
from PIL import Image,ImageFilter
import sys
import cv2 

class Preprocessor:
	def __init__(self):
		"""nothing"""
		
	def preprocess(self, jpgtxt):
		data = jpgtxt.split(',')[-1]
		encode_string = data.encode('ascii')
		decode_string = base64.b64decode(encode_string)
		decode_image = np.fromstring(decode_string, dtype=np.uint8)
		original_image = cv2.imdecode(decode_image, 1)
		cv2.imwrite('/home/l-ubuntus/Documents/code/html/app_conv_test/app/temp.jpg',original_image)
		g = open("temp.jpg", "wb")
		g.write(data)
		g.close()

		pic = Image.open("temp.jpg")
		M = np.array(pic) #now we have image data in numpy
		argv = '/home/l-ubuntus/Documents/code/html/app_conv_test/temp.jpg'
		im = Image.open(argv).convert('L')
		width = float(im.size[0])
		height = float(im.size[1])
		# creates white canvas of 28x28 pixels
		newImage = Image.new('L', (28, 28), (0))

		if width > height:  # check which dimension is bigger
			# Width is bigger. Width becomes 20 pixels.
			# resize height according to ratio width
			nheight = int(round((20.0 / width * height), 0))
			if (nheight == 0):  # rare case but minimum is 1 pixel
				nheight = 1
				# resize and sharpen
			img = im.resize((20, nheight), Image.ANTIALIAS).filter(
				ImageFilter.SHARPEN)
			# calculate horizontal position
			wtop = int(round(((28 - nheight) / 2), 0))
			newImage.paste(img, (4, wtop))  # paste resized image on white canvas
		else:
			# Height is bigger. Heigth becomes 20 pixels.
			# resize width according to ratio height
			nwidth = int(round((20.0 / height * width), 0))
			if (nwidth == 0):  # rare case but minimum is 1 pixel
				nwidth = 1
				# resize and sharpen
			img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(
				ImageFilter.SHARPEN)
			# caculate vertical pozition
			wleft = int(round(((28 - nwidth) / 2), 0))
			newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
		newImage.save("/home/l-ubuntus/Pictures/digits/digit.jpg")
		img = cv2.imread("/home/l-ubuntus/Pictures/digits/digit.jpg")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.asarray(img).reshape(-1)
		img = img/255.0
		return img

	