import numpy as np
import scipy.io.wavfile as sio
import PIL
from PIL import Image
import pyaudio
import wave
import sys
import matplotlib.pyplot as plt
import math
import contextlib


class ImageSound(object):
	def __init__(self, filename):
		if filename is None:
			print("Must pass in a file name to ImageSound.")
			sys.exit(-1)
		self.imageFile = filename

	def __d2xy__(self, m, d):
		n = 2 ** m

		x = 0
		y = 0
		t = d
		s = 1

		while ( s < n ):
			rx = ( ( t // 2 ) % 2 )
			if ( rx == 0 ):
				ry = ( t % 2 )
			else:
				ry = ( ( t ^ rx ) % 2 )
			x, y = self.__rot__( s, x, y, rx, ry )
			x = x + s * rx
			y = y + s * ry
			t = ( t // 4 )

			s = s * 2

		return x, y

	def __rot__(self, n, x, y, rx, ry):
		if ( ry == 0 ):
			#
			#  Reflect.
			#
			if ( rx == 1 ):
				x = n - 1 - x
				y = n - 1 - y
			#
			#  Flip.
			#
			t = x
			x = y
			y = t

		return x, y

	def find_order(self, size):
		# print("finding order of: ", size)
		order = 0

		while size > 1:
			size = int(size / 2)
			order += 1

		return order

	def resize(self, image, size, t = Image.NEAREST):
		return image.resize((size, size), t)

	def get_raw_values (self):
		image = Image.open(self.imageFile)
		width, height = image.size

		order = self.find_order(min(width, height))
		size = 2 ** order

		newImage = self.resize(image, size, t = Image.NEAREST)
		pix = newImage.load()

		triverse_order = np.empty((size, size))
		# triverse_value = np.empty((size * size, 3))
		triverse_value = np.empty((size*size, 1))

		for d in range ( 0, size * size ):
			x, y = self.__d2xy__( order, d )
			triverse_order[x,y] = d
			values = pix[x,y]
			values = (values[0] + values[1] + values[2]) / 3
			# values = 255 - values
			triverse_value[d] = values

		triverse_value = triverse_value.astype('uint8')

		return triverse_value

	def filterSound(self, fileName):
		print("filtering sound")

	def playImage(self, rate = 44100):
		raw_values = self.get_raw_values()
		newFileName = self.imageFile[:-3] + 'wav'

		sio.write(newFileName, rate, raw_values)


		CHUNK = 1024

		# wf = wave.open(sys.argv[1], 'rb')
		wf = wave.open(newFileName, 'rb')

		# instantiate PyAudio (1)
		p = pyaudio.PyAudio()

		# open stream (2)
		stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
		                channels=wf.getnchannels(),
		                rate=wf.getframerate(),
		                output=True)

		# read data
		data = wf.readframes(CHUNK)   # type(data) = <class 'bytes'>



		# play stream (3)
		while len(data) > 0:
		  stream.write(data)  # data is of type <class 'bytes'>
		  data = wf.readframes(CHUNK)


		# stop stream (4)
		stream.stop_stream()
		stream.close()

		# close PyAudio (5)
		p.terminate()


if ( __name__ == '__main__' ):
  filename = 'image1.jpg'
  raw_values = ImageSound(filename).get_raw_values()
  raw_values = raw_values.astype('uint8')
  rate = 44100
  sio.write('source1.wav', 44100, raw_values)
  # print("size = ", raw_values.shape)
  print(type(raw_values[0,0]))
  test = ImageSound(filename)
  test.playImage()