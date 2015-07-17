#!/usr/bin/python

import cv2
import csv
import os
import numpy as np
import math
import irtk
import argparse

from glob import glob
from sklearn import neighbors
from sklearn import ensemble
from sklearn import svm
import joblib
from joblib import Parallel, delayed

from lib.BundledSIFT import *

import sys
sys.path.insert( 1, "../commonlib" )
from fetal_anatomy import *

######################################################################

def process_file( f,
                  ga,
                  coordinates,
                  size,
                  classifier,
                  N,
                  DEBUG=False ):
    X = []
    Y = []

    scan_id = os.path.basename(f).split('.')[0]

    max_e = 0.64
    OFD = get_OFD(30.0,centile=50)

    mser = cv2.MSER( _delta=5,
                     _min_area=60,
                     _max_area=14400,
                     _max_variation=0.15,
                     _min_diversity=.1,
                     _max_evolution=200,
                     _area_threshold=1.01,
                     _min_margin=0.003,
                     _edge_blur_size=5)

    sift = cv2.SIFT( nfeatures=0,
                     nOctaveLayers=3,
                     contrastThreshold=0.04,
                     edgeThreshold=10,
                     sigma=0.8)
    siftExtractor = cv2.DescriptorExtractor_create("SIFT")

    box = np.array(map(float,coordinates.split(',')),dtype='float')
    size = np.array(map(float,size.split(',')),dtype='float')

    img = irtk.imread( f, dtype='float32', force_neurological=False )
    old_img = img.copy()

    ## Resample
    img = resampleOFD( img, ga )
    
    # Adjust coordinates and size
    box = img.WorldToImage( old_img.ImageToWorld(box[::-1]) )[::-1]

    box_size = size * old_img.header['pixelSize'][:3][::-1]/img.header['pixelSize'][:3][::-1]

    z0,y0,x0 = box.astype('int')
    d0,h0,w0 = box_size.astype('int')

    brain_center = (x0 + w0/2, y0 + h0/2)

    ## Contrast-stretch with saturation
    img = img.saturate(1,99).rescale().astype('uint8')

    for z in range(img.shape[0]):
        contours = mser.detect(img[z])
        keypoints = sift.detect(img[z])
        if keypoints is None or len(keypoints) == 0:
            continue
        (keypoints, descriptors) = siftExtractor.compute(img[z],keypoints)     

        for i,c in enumerate(contours):
            hist = np.zeros(N, dtype='float')
            ellipse = cv2.fitEllipse(np.array(map(lambda x:[x],
                                                  c),dtype='int32'))
            
            # filter by size
            if ( ellipse[1][0] > OFD
                 or ellipse[1][1] > OFD
                 or ellipse[1][0] < 0.5*OFD
                 or ellipse[1][1] < 0.5*OFD ) :
                continue

            # filter by eccentricity
            if math.sqrt(1-(np.min(ellipse[1])/np.max(ellipse[1]))**2) > max_e:
                continue

            distance = math.sqrt((ellipse[0][0]-brain_center[0])**2
                                 +(ellipse[0][1]-brain_center[1])**2)

            if max(w0,h0)/2 >= distance >= min(w0,h0)/8:
                continue            

            for k,d in zip(keypoints,descriptors):
                if is_in_ellipse(k.pt,ellipse):
                    c = classifier.kneighbors(d, return_distance=False)
                    hist[c] += 1

            # Normalize histogram
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm

            if distance > max(w0,h0)/4:
                if DEBUG: print 0
                X.append(hist)
                Y.append(0)
            else:
                if distance < min(w0,h0)/8 and z0 + d0/8 <= z <= z0+7*d0/8:
                    if DEBUG: print 1
                    X.append(hist)
                    Y.append(1)
                else:
                    continue
                   
            if DEBUG and Y[-1] == 1:
                img_color = cv2.cvtColor( img[z], cv2.cv.CV_GRAY2RGB )
                cv2.ellipse( img_color, (ellipse[0],
                                         (ellipse[1][0],ellipse[1][1]),
                                         ellipse[2]) , (0,0,255))
                for k_id,k in enumerate(keypoints):
                    if is_in_ellipse(k.pt,ellipse):
                        if Y[-1] == 1:
                            cv2.circle( img_color,
                                        (int(k.pt[0]),int(k.pt[1])),
                                        2,
                                        (0,255,0),
                                        -1)
                        else:
                            cv2.circle( img_color,
                                        (int(k.pt[0]),int(k.pt[1])),
                                        2,
                                        (0,0,255),
                                        -1)
                cv2.imwrite("debug/"+scan_id+'_'+str(z) + '_' +str(i) +'_'+str(k_id)+".png",img_color)

    return X,Y
        
######################################################################

parser = argparse.ArgumentParser(
    description='Learn MSER classifier using SIFT BOW.' )
parser.add_argument( '--training_patients' )
parser.add_argument( '--data_folder' )
parser.add_argument( '--ga_file' )
parser.add_argument( '--brainboxes' )
parser.add_argument( '--vocabulary' )
parser.add_argument( '--output' )
parser.add_argument( '--n_jobs', type=int, default=-1 )
parser.add_argument( '--debug', action="store_true", default=False )

args = parser.parse_args()

if args.debug and not os.path.exists( "debug" ):
    os.makedirs( "debug" )
    
vocabulary = open(args.vocabulary, 'rb')
voca = np.load(vocabulary)
classifier = neighbors.NearestNeighbors(1)
N = voca.shape[0] 
classifier.fit(voca)

f = open( args.training_patients, "r" )
patients = []
for p in f:
    patients.append(p.rstrip())
f.close()

reader = csv.reader( open( args.ga_file, "rb"), delimiter=" " )
all_ga = {}
for patient_id, ga in reader:
    all_ga[patient_id] = float(ga)
    
reader = csv.reader( open( args.brainboxes, "r" ),
                     delimiter='\t' )

training_patients = []
for patient_id, filename, cl, coordinates, size in reader:
    if patient_id not in patients:
        print "Skipping testing patient: " + patient_id
        continue
    training_patients.append( ( args.data_folder + '/' + filename,
                                all_ga[patient_id], coordinates, size ))

XY = Parallel(n_jobs=args.n_jobs)(delayed(process_file)(filename,ga,coordinates,
                                                        size, classifier,N,args.debug)
                                  for filename,ga,coordinates, size in training_patients )

print len(XY)

X = []
Y = []
for x,y in XY:
    X.extend(x)
    Y.extend(y)

print "RATIO = ", np.sum(Y), len(Y)
    
X = np.array(X,dtype='float')
Y = np.array(Y,dtype='float')

# classifier = ensemble.RandomForestClassifier( n_estimators=100,
#                                               n_jobs=10 )

classifier = svm.LinearSVC( dual=False )

classifier.fit(X, Y)
print classifier.score(X, Y)

joblib.dump(classifier, args.output)
