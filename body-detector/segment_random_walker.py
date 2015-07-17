#!/usr/bin/python

import numpy as np
import irtk
import scipy.ndimage as nd
from skimage import morphology
from skimage.segmentation import random_walker

import argparse

parser = argparse.ArgumentParser(
    description='' )
parser.add_argument( '--seg', type=str, required=True )
parser.add_argument( '--img', type=str, required=True )
parser.add_argument( '--output', type=str, required=True )
parser.add_argument( '--narrow_band', type=int, default=5 )
parser.add_argument( '--thorax', action="store_true", default=False )
parser.add_argument( '--debug', action="store_true", default=False )

args = parser.parse_args()

seg = irtk.imread( args.seg, dtype='int32', force_neurological=True )
img = irtk.imread( args.img, dtype='float32', force_neurological=True ).rescale(0,1000)

if args.thorax:
    seg[seg==4] = 4  # no liver

# crop
x_min,y_min,z_min,x_max,y_max,z_max = (seg > 0).bbox()
tmp_seg = seg[max(0,z_min-args.narrow_band-1):min(seg.shape[0],z_max+args.narrow_band+1+1),
              max(0,y_min-args.narrow_band-1):min(seg.shape[1],y_max+args.narrow_band+1+1),
              max(0,x_min-args.narrow_band-1):min(seg.shape[2],x_max+args.narrow_band+1+1)]
tmp_img = img[max(0,z_min-args.narrow_band-1):min(img.shape[0],z_max+args.narrow_band+1+1),
              max(0,y_min-args.narrow_band-1):min(img.shape[1],y_max+args.narrow_band+1+1),
              max(0,x_min-args.narrow_band-1):min(img.shape[2],x_max+args.narrow_band+1+1)]

background = (nd.binary_dilation( tmp_seg>0,
                                  structure=morphology.ball(args.narrow_band) ) == 0).astype('int32')

if args.thorax:
    tmp_seg[background>0] = 4
else:
    tmp_seg[background>0] = 5

if args.debug:
    debug_seg = tmp_seg.transform(target=img,interpolation='nearest')
    debug_seg[irtk.largest_connected_component(debug_seg==0)>0] = debug_seg.max()
    irtk.imwrite("debug_seg.nii.gz",debug_seg)
    irtk.imwrite("debug_background.nii.gz",debug_seg!=5)

tmp_img = tmp_img.rescale(-1,1)
labels = random_walker( tmp_img.view(np.ndarray),
                        tmp_seg.view(np.ndarray),
                        beta=1000,
                        mode='cg_mg',
                        return_full_prob=False )

header = tmp_img.get_header()
#header['dim'][3] = 5

if args.thorax:
    labels[labels==4] = 0
else:
    labels[labels==5] = 0

labels = irtk.Image(labels,header)

labels = labels.transform( target=img, interpolation='nearest' )

irtk.imwrite( args.output, labels )
