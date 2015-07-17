#!/usr/bin/python

import cv2
import os
import csv
import numpy as np
from math import cos,sin
import math
import irtk
import argparse

from glob import glob
from sklearn import neighbors
from sklearn import svm
import joblib
from joblib import Parallel, delayed

import ransac

import sys
sys.path.insert( 1, os.path.dirname(__file__)+"/../../commonlib" )
from fetal_anatomy import *

__all__ = [ "is_in_ellipse",
            "resampleOFD",
            "detect_mser",
            "ransac_ellipses" ]

######################################################################

def is_in_ellipse( (x,y), ((xe,ye),(we,he),theta)):
    theta = theta / 180 * np.pi
    u = cos(theta)*(x-xe)+ sin(theta)*(y-ye)
    v = -sin(theta)*(x-xe)+cos(theta)*(y-ye)

    # http://answers.opencv.org/question/497/extract-a-rotatedrect-area/
    # http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    # if theta < -45:
    #     tmp = we
    #     we = he
    #     he = we

    a = we/2
    b = he/2

    return (u/a)**2 + (v/b)**2 <= 1

def resampleOFD(img,ga,res0=1.0,interpolation='linear'):
    scale = get_OFD(ga,centile=50)/get_OFD(30.0,centile=50)
    img = img.resample2D( res0*scale, interpolation=interpolation )
    return img

def detect_mser( f,
                 ga,
                 vocabulary,
                 mser_detector,
                 output_folder="debug",
                 DEBUG=False ):

    max_e = 0.64
    OFD = get_OFD(30.0,centile=50) 
    BPD = get_BPD(30.0,centile=50)

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

    voca = np.load( open(vocabulary, 'rb') )
    nn_classifier = neighbors.NearestNeighbors(1,algorithm='kd_tree')
    N = voca.shape[0] 
    nn_classifier.fit(voca)

    classifier = joblib.load(mser_detector)
    
    img = irtk.imread( f,
                       dtype='float32',
                       force_neurological=False ).saturate(1,99).rescale()

    img = resampleOFD( img, ga ).astype('uint8')

    detected_centers = []
    detected_regions = []
    for z in range(img.shape[0]):
        
        # Extract MSER
        #print "extracting mser"
        contours = mser.detect(img[z])
        #print "mser done"
        
        if DEBUG:
            img_color = cv2.cvtColor( img[z], cv2.cv.CV_GRAY2RGB )
            for c in contours:
                ellipse = cv2.fitEllipse(np.array(map(lambda x:[x],
                                                  c),dtype='int32'))
                cv2.ellipse( img_color, (ellipse[0],
                                         (ellipse[1][0],ellipse[1][1]),
                                         ellipse[2]) , (0,0,255))

            cv2.imwrite(output_folder + "/" +str(z) + "_all_mser_.png",img_color )

            img_color_mser = cv2.cvtColor( img[z], cv2.cv.CV_GRAY2RGB )

        # Filter MSER
        selected_mser = []
        mask = np.zeros( (img.shape[1],img.shape[2]), dtype='uint8' )
        #print "fitting ellipses"
        for c in contours:
            ellipse = cv2.fitEllipse(np.reshape(c, (c.shape[0],1,2) ).astype('int32'))

            # filter by size
            if ( ellipse[1][0] > OFD
                 or ellipse[1][1] > OFD
                 or ellipse[1][0] < 0.5*OFD
                 or ellipse[1][1] < 0.5*OFD ) :
                continue

            # filter by eccentricity
            if math.sqrt(1-(np.min(ellipse[1])/np.max(ellipse[1]))**2) > max_e:
                continue

            cv2.ellipse( mask, ellipse, 255, -1 )
            selected_mser.append((c,ellipse))

        #print "ellipses done"
        if len(selected_mser) == 0:
            continue

        # Extract SIFT
        #print "extracting SIFT"
        keypoints = sift.detect(img[z],mask=mask)
        #print "SIFT done"
        if keypoints is None or len(keypoints) == 0:
            continue
        (keypoints, descriptors) = siftExtractor.compute(img[z],keypoints)

        words = nn_classifier.kneighbors(descriptors, return_distance=False)
        
        for i,(c,ellipse) in enumerate(selected_mser):
            # Compute histogram
            hist = np.zeros(N, dtype='float')
            for ki,k in enumerate(keypoints):
                if is_in_ellipse(k.pt,ellipse):
                    hist[words[ki]] += 1

            # Normalize histogram
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm

            cl = classifier.predict(hist)

            if DEBUG:
                if cl == 1:
                    opacity = 0.4
                    img_color = cv2.cvtColor( img[z], cv2.cv.CV_GRAY2RGB )
                    for p in c:
                        img_color[p[1],p[0],:] = (
                            (1-opacity)*img_color[p[1],p[0],:]
                            + opacity * np.array([0,255,0])
                            )
                    cv2.imwrite(output_folder + "/"+str(z) + '_' +str(i) +"_region.png",img_color)

                img_color = cv2.cvtColor( img[z], cv2.cv.CV_GRAY2RGB )
                
                cv2.ellipse( img_color, (ellipse[0],
                                         (ellipse[1][0],ellipse[1][1]),
                                         ellipse[2]) , (0,0,255))
                for k_id,k in enumerate(keypoints):
                    if is_in_ellipse(k.pt,ellipse):
                        if cl == 1:
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
                cv2.imwrite(output_folder + "/"+str(z) + '_' +str(i) +".png",img_color)

                cv2.ellipse( img_color_mser, (ellipse[0],
                                         (ellipse[1][0],ellipse[1][1]),
                                         ellipse[2]),
                             (0,255,0) if cl == 1 else (0,0,255) )

            if cl == 1:
                ellipse_center = [z,ellipse[0][1],ellipse[0][0]]
                detected_centers.append( ellipse_center )
                detected_regions.append( c )
            
        if DEBUG:
            cv2.imwrite(output_folder + "/"+str(z) + "_color_mser.png",img_color_mser)


    return np.array(detected_centers, dtype='int32'), np.array(detected_regions)
        
class BoxModel:
    def __init__(self,ofd,debug=False):
        self.debug = debug
        self.ofd = ofd
        
    def fit(self, data):
        mean = data[:,:3].mean(axis=0)
        return mean, np.array([0,0,1.0],dtype='float'), self.ofd
    
    def get_error( self, data, model):
        mean, u, ofd = model
        err_per_point = np.zeros(len(data),dtype='float')
        for i in range(len(data)):
            tmp = data[i,:3] - mean
            if np.linalg.norm(tmp) > ofd/2:
                err_per_point[i] = np.inf
            elif ( abs(data[i,3] - mean[0]) > ofd/2
                   or abs(data[i,4] - mean[0]) > ofd/2
                   or abs(data[i,5] - mean[1]) > ofd/2
                   or abs(data[i,6] - mean[1]) > ofd/2 ):
                err_per_point[i] = np.inf
            else:
                err_per_point[i] = np.linalg.norm(
                    tmp - np.dot(tmp,u)*u)
        
        return err_per_point

def ransac_ellipses( detected_centers,
                     detected_regions,
                     img,
                     ga,
                     nb_iterations=100,
                     debug=False ):
    """
    RANSAC is performed in mm (isotropic), but not in world coordinates
    """
    OFD = get_OFD(ga,centile=50)

    centers = []
    for center, region in zip(detected_centers,detected_regions):
        c = center[::-1]*img.header['pixelSize'][:3]
        r = np.hstack( (region, # region is xy in image coordinates
                        [[center[0]]]*region.shape[0]) )*img.header['pixelSize'][:3]
        centers.append([c[0], # x
                        c[1], # y
                        c[2], # z
                        r[:,0].min(),r[:,0].max(),
                        r[:,1].min(),r[:,1].max(),  
                        r[:,2].min(),r[:,2].max()])  

    centers = np.array(centers,dtype='float')

    model = BoxModel(OFD,debug=debug)

    # run RANSAC algorithm
    ransac_fit, inliers = ransac.ransac( centers,
                                         model,
                                         4, # minimum number of data values required to fit the model
                                         nb_iterations, # maximum number of iterations
                                         10.0, # threshold for determining when a data point fits a model
                                         4, # number of close data values required to assert that a model fits well to data
                                         debug=debug,
                                         return_all=True )

    if ransac_fit is None:
        print "RANSAC fiting failed"
        exit(1)

    (center, u, ofd) = ransac_fit

    center /= img.header['pixelSize'][:3]
    
    return img.ImageToWorld(center), inliers

