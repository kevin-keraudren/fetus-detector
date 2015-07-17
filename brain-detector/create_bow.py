#!/usr/bin/python

import numpy as np
import cv2
from sklearn import cluster# as skcluster
import os
import csv
import irtk
from glob import glob
import argparse

from scipy.stats.mstats import mquantiles
from joblib import Parallel, delayed

from lib.BundledSIFT import *

# Removing duplicate rows
# http://mail.scipy.org/pipermail/scipy-user/2011-December/031193.html

######################################################################

def process_file( f, ga, step=1, DEBUG=False ):
    print f
    img = irtk.imread( f, dtype='float32', force_neurological=False )
    
    ## Resample
    img = resampleOFD( img, ga )

    ## Contrast-stretch with saturation
    img = img.saturate(1,99).rescale().astype('uint8')

    detector = cv2.SIFT( nfeatures=0,
                         nOctaveLayers=3,
                         contrastThreshold=0.04,
                         edgeThreshold=10,
                         sigma=0.8)
    descriptorExtractor = cv2.DescriptorExtractor_create("SIFT")

    points = []
    for z in range(0,img.shape[0],step):
        keypoints = detector.detect(img[z])
        if keypoints is None or len(keypoints) == 0:
            continue
        (keypoints, descriptors) = descriptorExtractor.compute(img[z],keypoints)       
        unique_index= np.unique( descriptors.dot(np.random.rand(128)),
                                 return_index=True)[1]
        points.extend(descriptors[unique_index])

        ## For debugging purpose:
        if DEBUG:
            img_color = cv2.cvtColor( img[z].astype('uint8'), cv2.cv.CV_GRAY2RGB )
            for y,x in F.transpose():
                cv2.circle( img_color,
                            (int(x),int(y)),
                            2,
                            (0,0,255),
                            -1)
            cv2.imwrite( "/tmp/"
                         + os.path.basename(f.rstrip('.nii'))
                         + "_" + str(z)
                         +".png", img_color )

    points = np.array(points)
    unique_index= np.unique( points.dot(np.random.rand(128)),
                                 return_index=True)[1]
    return points[unique_index]

######################################################################


parser = argparse.ArgumentParser(
    description='Learn SIFT words for BOW classification.' )
parser.add_argument( '--training_patients' )
parser.add_argument( '--ga_file' )
parser.add_argument( '--data_folder' )
parser.add_argument( '--n_jobs', type=int, default=-1 )
parser.add_argument( '--step', type=int )
parser.add_argument( '--output' )
parser.add_argument( '--debug', action="store_true", default=False )

args = parser.parse_args()

reader = csv.reader( open( args.ga_file, "rb"), delimiter=" " )
all_ga = {}
for patient_id, ga in reader:
    all_ga[patient_id] = float(ga)
    
f = open( args.training_patients, "r" )
patients = []
for p in f:
    patients.append(p.rstrip())
f.close()
      
originals = glob(args.data_folder + "/*")

training_files = []
for f in originals:
    patient_id = os.path.basename(f).split('_')[0]
    if patient_id in patients:
        training_files.append( f )

training_files = training_files[::args.step]

print "Will now process " + str(len(training_files)) + " files from " + str(len(patients)) + " patients..."
        
bulk_points = Parallel(n_jobs=args.n_jobs)(delayed(process_file)(f,
                                                                 all_ga[os.path.basename(f).split('_')[0]],
                                                                 args.step) for f in training_files)

points = []
for D in bulk_points:
    points.extend(D)

print len(points)
points = np.array(points,dtype='float')

print points, points.shape

print "Starting K-means clustering..."

kmeans = cluster.MiniBatchKMeans( n_clusters=400,
                                  max_iter=1000,
                                  compute_labels=False )
kmeans.fit(points)

print kmeans.cluster_centers_[:,-10:]

vocabulary = open(args.output, 'wb')
np.save(vocabulary, kmeans.cluster_centers_ )

print "Done"
