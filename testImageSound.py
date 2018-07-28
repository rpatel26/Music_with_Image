from imageSound import ImageSound


if ( __name__ == '__main__' ):
  file1 = './sources/image1.jpg'
  file2 = './sources/image2.jpg'
  
  test1 = ImageSound(file1)
  test1.playImage()
  test1.playFilteredImage()

  # print()
  # test2 = ImageSound(file2)
  # test2.playImage()
  # test2.playFilteredImage()