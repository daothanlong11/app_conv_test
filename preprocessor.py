import pickle
import numpy as np
import os.path
import base64
from PIL import Image,ImageFilter
import sys
import cv2 

class Preprocessor:
	def __init__(self, fn="dataset.txt"):
		self.datafile = fn
		self.target_shape = [20,20]
		self.splitchar = "&"

		if os.path.isfile(self.datafile):
			with open(fn, 'r') as f:
				self.vocab = eval(f.readline())
				assert type(self.vocab) is list, "The first line of the file was not a vocab list."
		else:
			self.vocab = None

	def add_sample(self, data, index, vocab):
		assert self.vocab == vocab or self.vocab is None, "Vocab list has changed. You cannot append to this dataset"
		if self.vocab is None:
			with open(self.datafile, 'w') as f:
				v = str(vocab).replace("\n","")
				f.write(v)
		self.vocab = vocab
		n = self.preprocess(data)
		with open(self.datafile, "a") as f:
			n = str(n.tolist()).replace("\n","")
			f.write("\n" + str(self.vocab[index]) + self.splitchar + n)

	def load_sample(self, line_i = 0):
		X = None ; y = None
		if os.path.isfile(self.datafile):
			with open(self.datafile, 'r') as f:
				vocab = eval(f.readline())
				assert vocab == self.vocab, "The vocab for this datafile does not match the current vocab"

			with open(self.datafile, "r") as f:
				lines = f.readlines()
				X = np.asarray(eval(lines[line_i + 1].split(self.splitchar)[1]))
				y = lines[line_i + 1].split(self.splitchar)[0]
			return [X, y]

		else:
			raise ValueError("Could not find the datafile")

	def load_all(self):
		X = None ; y = None
		if os.path.isfile(self.datafile):
			with open(self.datafile, 'r') as f:
				vocab = eval(f.readline())
				assert vocab == self.vocab, "The vocab for this datafile does not match the current vocab"

			with open(self.datafile, "r") as f:
				lines = f.readlines()

				X = np.zeros((len(lines) - 1, self.target_shape[0]*self.target_shape[1]))
				y = []

				for i in range (len(lines)-1):
					X[i,:] = np.asarray(eval(lines[i + 1].split(self.splitchar)[1]))
					y.append(lines[i + 1].split(self.splitchar)[0])
				y = np.asarray(y)
			return [X, y]

		else:
			raise ValueError("Could not find the datafile")

	def preprocess(self, jpgtxt):
		# data = base64.decodestring(data)
		data = jpgtxt.split(',')[-1]
		data = base64.b64decode(data.encode('ascii'))

		g = open("temp.jpg", "wb")
		g.write(data)
		g.close()
		argv = 'D:\\code\\html\\app_conv_test\\temp.jpg'
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
		newImage.save("E:\\digits\\digit.png")
		img = cv2.imread("E:\\digits\\digit.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.asarray(img).reshape(-1)
		img = img/255.0
		return img

	