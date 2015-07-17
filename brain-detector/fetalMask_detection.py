#!/usr/bin/python

import sys
import numpy as np
import irtk

import os
from glob import glob

from lib.BundledSIFT import *
import scipy.ndimage as nd
import cv2
from skimage import morphology

import argparse
import warnings

sys.path.insert(1,os.path.dirname(__file__)+"/../commonlib")
from fetal_anatomy import *

parser = argparse.ArgumentParser(
    description='Slice-by-slice detection of fetal brain MRI (3D).' )
parser.add_argument( "filename", type=str )
parser.add_argument( "ga", type=float )
parser.add_argument( "output_mask", type=str )
parser.add_argument( '--vocabulary', type=str, default=None )
parser.add_argument( '--classifier', type=str, default=None )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--fold', type=str, default='0' )
args = parser.parse_args()
print args

filename = args.filename
ga = args.ga
output_mask = args.output_mask

output_dir = os.path.dirname( output_mask )
if output_dir != '' and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if output_dir == '':
    output_dir = '.'

if ( args.vocabulary is None or args.classifier is None ):
    print "Setting default vocabulary and default classifier"
    vocabulary = "/vol/biomedic/users/kpk09/github/example-motion-correction/model/vocabulary_"+args.fold+".npy"
    mser_detector = "/vol/biomedic/users/kpk09/github/example-motion-correction/model/mser_detector_"+args.fold+"_linearSVM"
else:
    vocabulary = args.vocabulary
    mser_detector = args.classifier
    
print"Detect MSER regions"
detections = []

img = irtk.imread(filename, dtype="float32", force_neurological=False).saturate(1,99).rescale()

if img.header['pixelSize'][2] > 2.0*img.header['pixelSize'][0]:
    warnings.warn( "This methods proceeds slice-by-slice and has poorer " +
                   "performance on thick slices due to the lower number of " +
                   "slices." )
    
img_resampled = resampleOFD( img, ga, interpolation='nearest' )

detected_centers, detected_regions = detect_mser( filename,
                                                  ga,
                                                  vocabulary,
                                                  mser_detector,
                                                  DEBUG=args.debug,
                                                  output_folder=output_dir )

print "Fit box with RANSAC"
center, selection = ransac_ellipses( detected_centers,
                                     detected_regions,
                                     img_resampled,
                                     ga,
                                     nb_iterations=1000,
                                     debug=args.debug)

selected_centers = detected_centers[selection]
selected_regions = detected_regions[selection]

print "initial mask"

mask = irtk.zeros(img_resampled.get_header(), dtype='uint8')

# ellipse mask
ellipse_mask = irtk.zeros(img_resampled.get_header(), dtype='uint8')

for c,r in zip(selected_centers,selected_regions):
    mask[c[0],r[:,1],r[:,0]] = 1

    ellipse = cv2.fitEllipse(np.reshape(r, (r.shape[0],1,2) ).astype('int32'))
    tmp_img = np.zeros( (ellipse_mask.shape[1],ellipse_mask.shape[2]), dtype='uint8' )
    cv2.ellipse( tmp_img, (ellipse[0],
                           (ellipse[1][0],ellipse[1][1]),
                           ellipse[2]) , 1, thickness=-1)
    ellipse_mask[c[0]][tmp_img > 0] = 1

mask[ellipse_mask == 1] = 1

mask = mask.transform(target=img, interpolation='nearest' )

#irtk.imwrite("mask.nii.gz",mask)

# fill holes, close and dilate
disk_close = morphology.disk( 5 )
disk_dilate = morphology.disk( 2 )
for z in xrange(mask.shape[0]):
    mask[z] = nd.binary_fill_holes( mask[z] )
    mask[z] = nd.binary_closing( mask[z], disk_close )
    mask[z] = nd.binary_dilation( mask[z], disk_dilate )

neg_mask = np.ones(mask.shape, dtype='uint8')*2

x,y,z = img.WorldToImage(center)

ofd = get_OFD(ga,centile=95)

w = h = int(round( ofd / img.header['pixelSize'][0]))
d = int(round( ofd / img.header['pixelSize'][2]))

neg_mask[max(0,z-d/2):min(img.shape[0],z+d/2+1),
         max(0,y-h/2):min(img.shape[1],y+h/2+1),
         max(0,x-w/2):min(img.shape[2],x+w/2+1)] = 0

mask[neg_mask>0] = 2

irtk.imwrite(output_mask, mask )
