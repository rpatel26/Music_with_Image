import numpy as np

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

def d2xy_test ( ):
  order = 3
  size = 2 ** order

  triverse_order = np.empty((size, size))

  for d in range ( 0, size * size ):
    x, y = d2xy ( order, d )
    triverse_order[x,y] = d


  return triverse_order

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
  t = d2xy_test()
  print(t)
