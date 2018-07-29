from imageSound import ImageSound
import argparse


if ( __name__ == '__main__' ):
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', "--file", 
		help = "Fill path of the image source")
	args = parser.parse_args()

	if args.file is None:
		file1 = './sources/image1.jpg'
	else:
		file1 = args.file

	file2 = './sources/image2.jpg'

	test1 = ImageSound(file1)
	test1.playImage()
	test1.playFilteredImage()

	# print()
	# test2 = ImageSound(file2)
	# test2.playImage()
	# test2.playFilteredImage()