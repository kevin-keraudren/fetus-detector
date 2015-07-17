#!/usr/bin/python

import irtk
import numpy as np
import scipy.ndimage as nd

from glob import glob
import os

import joblib

import sys
sys.path.append( "../lib" )
from libdetector import *

from sklearn.decomposition import PCA

def get_coordinates( f ):
    seg = irtk.imread( f )
    u,v,w = get_orientation_training(seg)

    M = np.array( [u,v,w], dtype='float32' ) # Change of basis matrix
    
    heart =  np.array(nd.center_of_mass( (seg == 5).view(np.ndarray) ),
                      dtype='float32')
    brain =  np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                      dtype='float32')
    left_lung =  np.array(nd.center_of_mass( (seg == 3).view(np.ndarray) ),
                          dtype='float32')
    right_lung =  np.array(nd.center_of_mass( (seg == 4).view(np.ndarray) ),
                           dtype='float32')
    liver =  np.array(nd.center_of_mass( (seg == 8).view(np.ndarray) ),
                      dtype='float32')

    # centering and orient
    brain = np.dot( M, brain - heart)
    left_lung = np.dot( M, left_lung - heart)
    right_lung = np.dot( M, right_lung - heart)
    liver = np.dot( M, liver - heart)

    return np.array( [brain, left_lung, right_lung, liver], dtype='float32' ).flatten()
    
all_files = glob( "/vol/bitbucket/kpk09/detector/data_resampled_weight/*_seg.nii.gz" )

all_coordinates = map( get_coordinates, all_files )

# whiten?
pca = PCA()
pca.fit( all_coordinates )

print pca.explained_variance_ratio_

if not os.path.exists("output"):
    os.makedirs("output")
    
joblib.dump(pca,"output/pca_landmarks")
