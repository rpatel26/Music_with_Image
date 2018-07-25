import numpy as np
import scipy.io.wavfile as sio
import PIL
from PIL import Image
from testHilbert import ImageSound


def d2xy ( m, d ):

#*****************************************************************************80
#
## D2XY converts a 1D Hilbert coordinate to a 2D Cartesian coordinate.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 January 2016
#
#  Parameters:
#
#    Input, integer M, the index of the Hilbert curve.
#    The number of cells is N=2^M.
#    0 < M.
#
#    Input, integer D, the Hilbert coordinate of the cell.
#    0 <= D < N * N.
#
#    Output, integer X, Y, the Cartesian coordinates of the cell.
#    0 <= X, Y < N.
#
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
    x, y = rot ( s, x, y, rx, ry )
    x = x + s * rx
    y = y + s * ry
    t = ( t // 4 )

    s = s * 2

  return x, y

def find_order(size):
  # print("finding order of: ", size)
  order = 0

  while size > 1:
    size = int(size / 2)
    # print("size = ", size)

    order += 1

  return order

def resize(image, size, t = Image.NEAREST):
  return image.resize((size, size), t)

def d2xy_test (image):
  width, height = image.size

  order = find_order(min(width, height))
  size = 2 ** order

  newImage = resize(image, size, t = Image.NEAREST)
  pix = newImage.load()

  # print("image size = (", height, ",", width, ")")
  # print("order = ", order)
  # print("2 ^ order = ", size)

  triverse_order = np.empty((size, size))
  # triverse_value = np.empty((size * size, 3))
  triverse_value = np.empty((size*size, 1))

  for d in range ( 0, size * size ):
    x, y = d2xy ( order, d )
    triverse_order[x,y] = d
    values = pix[x,y]
    values = (values[0] + values[1] + values[2]) / 3
    # values = 255 - values
    triverse_value[d] = values

  return triverse_value

def rot ( n, x, y, rx, ry ):

#*****************************************************************************80
#
## ROT rotates and flips a quadrant appropriately.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 January 2016
#
#  Parameters:
#
#    Input, integer N, the length of a side of the square.  
#    N must be a power of 2.
#
#    Input/output, integer X, Y, the coordinates of a point.
#
#    Input, integer RX, RY, ???
#
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



if ( __name__ == '__main__' ):
  # img = sio.imread('image1.jpg') # (168,300,3)
  img = Image.open('image1.jpg')
  raw_values = d2xy_test(img)
  raw_values = raw_values.astype('uint8')
  rate = 44100
  sio.write('source1.wav', 44100, raw_values)
  # print("size = ", raw_values.shape)
  # print(type(raw_values[0,0]))
  temp = ImageSound()
  temp2 = temp.get_raw_values(img)



# imageFile = 'image1.jpg'

# img = Image.open(imageFile)
# img_width, img_height = img.size
# pix = img.load()

# # adjust the width and height to my needs
# width = 128
# height = 128

# # using nearest neighbour
# img2 = img.resize((width, height), Image.NEAREST)

# # linear interpolation in a 2x2 environment
# img3 = img.resize((width, height), Image.BILINEAR)

# # cubic splin interpolation in a 4x4 environment
# img4 = img.resize((width, height), Image.BICUBIC)

# # best down-sizing filter 
# img5 = img.resize((width, height), Image.ANTIALIAS)




import pyaudio
import wave
import sys

CHUNK = 1024

# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)

# wf = wave.open(sys.argv[1], 'rb')
wf = wave.open('filtered1.wav', 'rb')

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
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib

fname = 'source1.wav'
outname = 'filtered1.wav'

cutOffFrequency = 400.0

# from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

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

with contextlib.closing(wave.open(fname,'rb')) as spf:
    sampleRate = spf.getframerate()
    ampWidth = spf.getsampwidth()
    nChannels = spf.getnchannels()
    nFrames = spf.getnframes()

    # Extract Raw Audio from multi-channel Wav File
    signal = spf.readframes(nFrames*nChannels)
    spf.close()
    channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

    # get window size
    # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
    freqRatio = (cutOffFrequency/sampleRate)
    N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

    # Use moviung average (only on first channel)
    filtered = running_mean(channels[0], N).astype(channels.dtype)

    wav_file = wave.open(outname, "w")
    wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()
'''