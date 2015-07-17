#!/usr/bin/python

import irtk
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(__file__)+"/lib")
sys.path.append(os.path.dirname(__file__)+"/../commonlib")
sys.path.append(os.path.dirname(__file__)+"/../autocontext-segmentation/")

import scipy.ndimage as nd

import joblib
from joblib import Parallel, delayed
import cPickle as pickle

import argparse

from skimage.feature import peak_local_max
import subprocess

from libdetector import *
from helpers import *
from fetal_anatomy import *

from integralforest_aux import gdt_smooth

parser = argparse.ArgumentParser(
        description='' )
# required parameters
parser.add_argument( "--input", type=str, required=True, action=PathExists )
parser.add_argument( "--detector", type=str, required=True, action=PathExists )
parser.add_argument( "--shape_model", type=str, required=True, action=PathExists )
parser.add_argument( '--template', type=str, required=True, action=PathExists )
parser.add_argument( "--output", type=str, required=True )
# parameters with default values
parser.add_argument( '--ga', type=float, default=0 )
parser.add_argument( '--brain_center', type=float, nargs=3 )
parser.add_argument( '--verbose', action="store_true", default=False )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--n_jobs', type=int, default=20 )
parser.add_argument( '--chunk_size', type=int, default=int(3e6) )
parser.add_argument( '-l', type=float, default=0.5 )
parser.add_argument( '--theta', type=int, default=90 )
args = parser.parse_args()

print args

output_dir = os.path.dirname(args.output)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

template = irtk.imread(args.template,force_neurological=True)
img = irtk.imread(args.input,force_neurological=True)

if args.ga > 0:
    scale = get_CRL(args.ga)/get_CRL(30.0)
    resized_img = img.resample( 1.0*scale, interpolation='bspline' ).rescale(0,1000)
    resized_input = output_dir + "/resized_" + os.path.basename(args.input)
    irtk.imwrite( resized_input, resized_img )

    heart_center = resized_img.WorldToImage([0,0,0])[::-1]#np.array(resized_img.shape,dtype='float32')/2
    if not args.brain_center:
        brain_center = heart_center + np.array([0,0,100])
    else:
        brain_center = np.array(args.brain_center,dtype='float32')

else:
    resized_input = args.input
    heart_center = np.array(img.shape,dtype='float32')/2
    if not args.brain_center:
        brain_center = heart_center + np.array([0,0,100])
    else:
        brain_center = np.array(args.brain_center,dtype='float32')

cmd = [ 'python', os.path.dirname(__file__)+'/predict2.py',
        '--input', resized_input,
        '--detector', args.detector,
        '--n_jobs', args.n_jobs,
        '--chunk_size', args.chunk_size,
        '--shape_model', args.shape_model,
        '--theta', str(360),
        '--aligned',
        '-l', str(args.l),
        '--padding', str(0),
        '--output', output_dir,
        '--back_proj',
        '--brain_center'] + map(str,brain_center) + \
        ['--heart_center'] + map(str,heart_center)
        
if args.verbose:
    cmd.append('--verbose')

if args.debug:
    cmd.append('--debug')
        
cmd = map(str,cmd)
if args.debug or args.verbose:
    print " ".join(cmd)
        
p = subprocess.Popen( cmd,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE )
output, err = p.communicate()
rc = p.returncode

if args.debug:
    print output
    print err

# left_lung = irtk.imread( output_dir+"/back_proj_left_lung.nii.gz" )
# right_lung = irtk.imread( output_dir+"/back_proj_right_lung.nii.gz" )

# both_lungs = left_lung + right_lung
# l = 100.0
# v = 50.0
# includeEDT = True

# # both_lungs = gdt_smooth( both_lungs,
# #                          img,
# #                          l=l,
# #                          v=v,
# #                          includeEDT=includeEDT )

# both_lungs = nd.gaussian_filter(both_lungs,2.0)
# both_lungs /= both_lungs.max()

# both_lungs = gdt_smooth( both_lungs,
#                          img,
#                          l=l,
#                          v=v,
#                          includeEDT=includeEDT )

# header = left_lung.get_header()
# header['dim'][3] = 2
# proba = irtk.zeros( header, dtype='float32' )
# proba[1] = both_lungs
# proba[0] = 1.0 - both_lungs
# irtk.imwrite(output_dir+"/proba_both_lungs.nii.gz",proba)

# CRF segmentation
cmd = [ 'python', os.path.dirname(__file__)+'/segment_reconstruction.py',
        '--proba', output_dir+"/proba.nii.gz",
        #'--proba', output_dir+"/proba_both_lungs.nii.gz",
        '--img', output_dir+"/img.nii.gz",
        '--seg', output_dir+"/final_seg.nii.gz",
        '--output', output_dir+"/crf.nii.gz",
        '--remove_small_objects', str(20),
        '--dilate', str(3),
        '-l', str(30),
        '--sigma', str(10),
        '--select',
        #'--hull'
        ]

cmd = map(str,cmd)
if args.debug or args.verbose:
    print " ".join(cmd)
    
p = subprocess.Popen( cmd,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE )
output, err = p.communicate()
rc = p.returncode

if args.debug:
    print output
    print err

# take mask and isolate lungs

mask = irtk.imread( output_dir+"/crf.nii.gz" ).transform(target=img,interpolation='nearest')
irtk.imwrite( args.output, mask )
    
