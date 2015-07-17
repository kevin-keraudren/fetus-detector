#!/usr/bin/python

import numpy as np
import irtk
import scipy.ndimage as nd
from skimage import morphology

import argparse

parser = argparse.ArgumentParser(
    description='' )
parser.add_argument( '--seg', type=str, required=True )
parser.add_argument( '--img', type=str, required=True )
parser.add_argument( '--output', type=str, required=True )
parser.add_argument( '--narrow_band', type=int, default=5 )
parser.add_argument( '--debug', action="store_true", default=False )

args = parser.parse_args()

seg = irtk.imread( args.seg, dtype='int32', force_neurological=True )
img = irtk.imread( args.img, dtype='float32', force_neurological=True ).rescale(0,1000)

res = irtk.zeros( seg.get_header(), dtype='uint8' )

ball = morphology.ball( args.narrow_band )

nb_labels = 5

# for i in range(1,5):
#     tmp_seg = (seg==i).astype('int32')
#     # crop
#     x_min,y_min,z_min,x_max,y_max,z_max = (tmp_seg).bbox()
#     mask = tmp_seg[max(0,z_min-2*args.narrow_band):min(seg.shape[0],z_max+2*args.narrow_band+1),
#                       max(0,y_min-2*args.narrow_band):min(seg.shape[1],y_max+2*args.narrow_band+1),
#                       max(0,x_min-2*args.narrow_band):min(seg.shape[2],x_max+2*args.narrow_band+1)]
#     tmp_img = img[max(0,z_min-2*args.narrow_band):min(img.shape[0],z_max+2*args.narrow_band+1),
#                   max(0,y_min-2*args.narrow_band):min(img.shape[1],y_max+2*args.narrow_band+1),
#                   max(0,x_min-2*args.narrow_band):min(img.shape[2],x_max+2*args.narrow_band+1)]

#     background = (nd.binary_dilation( mask, structure=ball ) == 0).astype('int32')
#     background = irtk.largest_connected_component( background )
    
#     mask[background==1] = 2

#     irtk.imwrite( "mask.nii.gz", mask )
#     tmp_seg = irtk.graphcut( tmp_img, mask,
#                          sigma=100.0 )

#     tmp_seg = irtk.largest_connected_component( tmp_seg )
#     tmp_seg = tmp_seg.transform( target=seg, interpolation='nearest' )

#     res[tmp_seg==1] = i

# irtk.imwrite( args.output, res )

# crop
x_min,y_min,z_min,x_max,y_max,z_max = (seg > 0).bbox()
tmp_seg = seg[max(0,z_min-2*args.narrow_band):min(seg.shape[0],z_max+2*args.narrow_band+1),
           max(0,y_min-2*args.narrow_band):min(seg.shape[1],y_max+2*args.narrow_band+1),
           max(0,x_min-2*args.narrow_band):min(seg.shape[2],x_max+2*args.narrow_band+1)]
tmp_img = img[max(0,z_min-2*args.narrow_band):min(img.shape[0],z_max+2*args.narrow_band+1),
              max(0,y_min-2*args.narrow_band):min(img.shape[1],y_max+2*args.narrow_band+1),
              max(0,x_min-2*args.narrow_band):min(img.shape[2],x_max+2*args.narrow_band+1)]

for i in range(1,5):
    tmp_seg[nd.binary_closing(tmp_seg==i,
                              structure=morphology.ball(2))>0] = i

background = (nd.binary_dilation( tmp_seg>0, structure=ball ) == 0).astype('int32')

header = tmp_img.get_header()
header['dim'][3] = 5
proba = irtk.zeros( header, dtype='float32' )

proba[0][tmp_seg==0] = 1.0

for i in range(1,5):
    proba[i][tmp_seg==i] = 1.0

irtk.imwrite( "proba.nii.gz", proba )

print proba.header, proba.shape

tmp_seg = irtk.crf( tmp_img,
                    tmp_seg,
                    proba,
                    l=4.0,
                    sigma=50 )

res = tmp_seg.transform( target=seg, interpolation='nearest' )

print np.unique(res)

irtk.imwrite( args.output, res )
