import numpy as np
import irtk

def get_center_brain_detection(f,world=True):
    input_mask = irtk.imread( f, force_neurological=False )
    mask = irtk.ones( input_mask.get_header(), dtype='uint8' )
    mask[input_mask == 2] = 0
    x_min, y_min, z_min, x_max, y_max, z_max = map( float, mask.bbox() )
    center = [ x_min + (x_max-x_min)/2,
               y_min + (y_max-y_min)/2,
               z_min + (z_max-z_min)/2 ]
    if not world:
        center = np.array(center,dtype='float64')[::-1]
        return center
    else:
        center = input_mask.ImageToWorld(center)
        return center

def get_box_center((z,y,x),(d,h,w),f):
    img = irtk.imread( f, empty=True, force_neurological=False )
    center = np.array( [ x+w/2,
                         y+h/2,
                         z+d/2 ], dtype='float' )
    center = img.ImageToWorld(center)
    return center
