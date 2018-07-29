import numpy as np
import scipy.io.wavfile as sio
# import PIL
from PIL import Image
import pyaudio
import wave
import sys
import os
import math
import contextlib

class ImageSound(object):

	'''
	Name: __init__()
	Description: constructor of the ImageSound class.
	Parameters:
		filename -- full path of the image to be played
	Return Values: none
	'''
	def __init__(self, filename = None):
		if filename is None:
			print("Must pass in a file name to ImageSound.")
			sys.exit(-1)
		print("ImageFile = ", filename)
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

	'''
	Name: filterSound()
	Description: filter sound for the raw image
	Parameters: 
		cuttOff -- cutt off frequency 
	Return Values: none
	'''
	def filterSound(self, cuttOff = 400.0):
		fname = self.getWAVfromImg()
		outname = self.getOutFileName()
		cutOffFrequency = cuttOff

		with contextlib.closing(wave.open(fname,'rb')) as spf:
			sampleRate = spf.getframerate()
			ampWidth = spf.getsampwidth()
			nChannels = spf.getnchannels()
			nFrames = spf.getnframes()

			# Extract Raw Audio from multi-channel Wav File
			signal = spf.readframes(nFrames*nChannels)
			spf.close()
			channels = self.__interpret_wav(signal, nFrames, nChannels, ampWidth, True)

			# get window size
			# from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
			freqRatio = (cutOffFrequency/sampleRate)
			N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

			# Use moviung average (only on first channel)
			filtered = self.__running_mean(channels[0], N).astype(channels.dtype)

			wav_file = wave.open(outname, "w")
			wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
			wav_file.writeframes(filtered.tobytes('C'))
			wav_file.close()

		return outname

	'''
	Name: findOrder
	Description: find the order of the Hilber curve
	Parameters: 
		size -- max size of the image
	Return Values:
		order -- order of the Hilber curve
	'''
	def findOrder(self, size):
		# print("finding order of: ", size)
		order = 0

		while size > 1:
			size = int(size / 2)
			order += 1

		return order

	'''
	Name: getOutFileName()
	Description: generates the full path name for the filtered sound file
	Parameters: none
	Return Values:
		newFileName -- full path name for the filtered sound file
	'''
	def getOutFileName(self):
		s = '/'
		path = os.path.normpath(self.imageFile)
		path_array = path.split(os.sep)
		newFileName = 'filtered_' + path_array[-1][:-3] + 'wav'
		path_array[-1] = newFileName
		newFileName = s.join(path_array)

		return newFileName

	'''
	Name: getRawValues()
	Description: get the raw values from the image file to be played
	Parameters: none
	Return Values: none
	'''
	def getRawValues(self):
		image = Image.open(self.imageFile)
		width, height = image.size

		order = self.findOrder(min(width, height))
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

	'''
	Name: getWAVformImg()
	Description: generate the WAV file from the image file
	Parameters:
		filename -- full path name of the image file to play
		rate -- sampling rate for the WAV file
	Return Values:
		newFileName -- full path name of the waver file
	'''
	def getWAVfromImg(self, filename = None, rate = 44100):
		if filename is None:
			filename = self.imageFile

		raw_values = self.getRawValues()
		newFileName = filename[:-3] + 'wav'
		sio.write(newFileName, rate, raw_values)

		return newFileName

	def __interpret_wav(self, raw_bytes, n_frames, n_channels, sample_width, interleaved = True):
		if sample_width == 1:
			dtype = np.uint8 # unsigned char
		elif sample_width == 2:
			dtype = np.int16 # signed 2-byte short
		else:
			raise ValueError("Only supports 8 and 16 bit audio formats.")

		channels = np.fromstring(raw_bytes, dtype=dtype)

		if interleaved:
			# channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
			channels.shape = (n_frames, n_channels)
			channels = channels.T
		else:
			# channels are not interleaved. All samples from channel M occur before all samples from channel M-1
			channels.shape = (n_channels, n_frames)

		return channels

	'''
	Name: __play()
	Description: Plays any WAV files
	Parameters:
		filename -- full path of the WAV file to paly
	Return Values: none
	'''
	def __play(self, filename):
		CHUNK = 1024

		# wf = wave.open(sys.argv[1], 'rb')
		wf = wave.open(filename, 'rb')

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

	'''
	Name: playImage()
	Description: plays the image file without any filters
	Parameters:
		filename -- full path of the image file to play 
		rate -- sampling rate for the audio 
	Return Values: none
	'''
	def playImage(self, filename = None, rate = 44100):
		newFileName = self.getWAVfromImg(filename, rate)
		print("playing raw image")
		self.__play(newFileName)

	'''
	Name: playFilteredImage()
	Description: plays the image file with filtered sound
	Parameters: none
	Return Values: none
	'''
	def playFilteredImage(self):
		fName = self.filterSound()
		print("playing filtered image")
		self.__play(fName)

	'''
	Name: resize()
	Description: resize any image to a square image
	Parameters:
		image -- image to resize
		size -- size of the image
		t -- type of algorithm
	Return Values: resized image
	'''
	def resize(self, image, size, t = Image.NEAREST):
		return image.resize((size, size), t)

	def __running_mean(self, x, windowSize):
		cumsum = np.cumsum(np.insert(x, 0, 0)) 
		return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize	