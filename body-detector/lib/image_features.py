import numpy as np

from lib._image_features import get_block_means

def integral_image( img ):
    """
    From Wikipedia:
    A summed area table is a data structure and algorithm for quickly and
    efficiently generating the sum of values in a rectangular subset of a
    grid. In the image processing domain, it is also known as an integral image.
    """
    return np.cumsum(np.cumsum(np.cumsum(img,2),1),0).astype('float32')

def get_block_comparisons( sat, coords,
                           offsets1, sizes1,
                           offsets2, sizes2 ):
    return ( get_block_means( sat,
                              coords,
                              offsets1,
                              sizes1 ) > get_block_means( sat,
                                                          coords,
                                                          offsets2,
                                                          sizes2 ) )
