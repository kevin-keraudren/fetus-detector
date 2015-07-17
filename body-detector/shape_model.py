#!/usr/bin/python

import irtk
import numpy as np
import scipy.ndimage as nd

from skimage import morphology

from glob import glob
import os

import joblib
from joblib import Parallel, delayed

import sys
sys.path.append( "lib" )
from libdetector import *

import cPickle as pickle

import argparse

def get_coordinates( f ):
    seg = irtk.imread( f, force_neurological=True )
    u,v,w = get_orientation_training(seg)

    # http://fr.wikipedia.org/wiki/Matrice_de_passage
    M = np.array( [w,v,u], dtype='float32' ) # Change of basis matrix

    print "DET:", np.linalg.det(M)
    
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

    d = np.linalg.norm(brain-heart)
    
    # centering and orient
    left_lung = np.dot( M, left_lung - heart)
    right_lung = np.dot( M, right_lung - heart)
    liver = np.dot( M, liver - heart)

    return [left_lung, right_lung, liver, d]

def get_coordinates2( f ):
    seg = irtk.imread( f, force_neurological=True )
    u,v,w = get_orientation_training(seg)

    # http://fr.wikipedia.org/wiki/Matrice_de_passage
    M = np.array( [w,v,u], dtype='float32' ) # Change of basis matrix

    print "DET:", np.linalg.det(M)

    heart_center =  np.array(nd.center_of_mass( (seg == 5).view(np.ndarray) ),
                      dtype='float32')
    brain_center =  np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                      dtype='float32')
    
    heart =  np.argwhere( (seg == 5).view(np.ndarray) ).astype('float32')
    left_lung =  np.argwhere( (seg == 3).view(np.ndarray) ).astype('float32')
    right_lung = np.argwhere( (seg == 4).view(np.ndarray) ).astype('float32')
    liver = np.argwhere( (seg == 8).view(np.ndarray) ).astype('float32')

    d = np.linalg.norm(brain_center-heart,axis=1)
    
    # centering and orient
    left_lung = np.transpose( np.dot( M, np.transpose(left_lung - heart_center)))
    right_lung = np.transpose( np.dot( M, np.transpose(right_lung - heart_center)))
    liver = np.transpose( np.dot( M, np.transpose(liver - heart_center)))
    heart = np.transpose( np.dot( M, np.transpose(heart - heart_center)))

    return [left_lung, right_lung, liver, d, heart]

def show_proba(mean,cov,header):
    res = irtk.zeros(header,dtype='float32')
    offset = np.array(res.shape)/2
    Z, Y, X = np.mgrid[ 0:res.shape[0],
                        0:res.shape[1],
                        0:res.shape[2] ]
    ZYX = np.vstack([Z.ravel(),Y.ravel(),X.ravel()]).astype('int32').transpose() 
    proba = mgaussian( ZYX-offset, mean, cov )
    res[ZYX[:,0],
        ZYX[:,1],
        ZYX[:,2]] = proba
    return res

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( "--seg", type=str, nargs='+' )
parser.add_argument( "--output_folder", type=str )
parser.add_argument( '--n_jobs', type=int, default=20 )
parser.add_argument( '--debug', action="store_true", default=False )
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

all_coordinates = Parallel(n_jobs=args.n_jobs)(delayed(get_coordinates2)(f)
                             for f in args.seg )

all_left_lungs = []
all_right_lungs = []
all_livers = []
all_ds = []
all_hearts = []
for left_lung, right_lung, liver, d, heart in all_coordinates:
    print left_lung.shape
    all_left_lungs.extend(left_lung)
    all_right_lungs.extend(right_lung)
    all_livers.extend(liver)
    all_ds.extend(d)
    all_hearts.extend(heart)

all_left_lungs = np.array( all_left_lungs, dtype='float64' )
all_right_lungs = np.array( all_right_lungs, dtype='float64' )
all_livers = np.array( all_livers, dtype='float64' )
all_ds = np.array( all_ds, dtype='float64' )
all_hearts = np.array( all_hearts, dtype='float64' )

mean_left_lung = np.mean( all_left_lungs, axis=0 )
mean_right_lung = np.mean( all_right_lungs, axis=0 )
mean_liver = np.mean( all_livers, axis=0 )
mean_brain = all_ds.mean()

# see sklearn.covariance.EmpiricalCovariance
cov_heart = np.cov( all_hearts.T )
cov_left_lung = np.cov( all_left_lungs.T )
cov_right_lung = np.cov( all_right_lungs.T  )
cov_liver = np.cov( all_livers.T )
sigma_brain = all_ds.std()

model = dict( cov_heart = cov_heart,
              cov_left_lung=cov_left_lung,
              cov_right_lung=cov_right_lung,
              cov_liver=cov_liver,
              sigma_brain=sigma_brain,
              mean_left_lung=mean_left_lung,
              mean_right_lung=mean_right_lung,
              mean_liver=mean_liver,
              mean_brain=mean_brain )

print model

pickle.dump( model, open(args.output_folder+"/shape_model.pk",'wb') )

if args.debug:
    average = irtk.imread("/vol/medic02/users/kpk09/gitlab/fetus-detector/body-detector/notebooks/tmp/new_average_heart_center.nii.gz",force_neurological=False)
    left_lung_prior = show_proba(model['mean_left_lung'],
                                 model['cov_left_lung'],
                                 irtk.Image(average).get_header())
    right_lung_prior = show_proba(model['mean_right_lung'],
                                  model['cov_right_lung'],
                                  irtk.Image(average).get_header())
    liver_prior = show_proba(model['mean_liver'],
                             model['cov_liver'],
                             irtk.Image(average).get_header())
    
    irtk.imwrite("left_lung_prior.nii.gz",left_lung_prior)
    irtk.imwrite("right_lung_prior.nii.gz",right_lung_prior)
    irtk.imwrite("liver_prior.nii.gz",liver_prior)

    all_coordinates = Parallel(n_jobs=args.n_jobs)(delayed(get_coordinates)(f)
                             for f in args.seg )
    scores = []
    for left_lung, right_lung, liver, d in all_coordinates:
        proba_left_lung = mgaussian( left_lung,
                                     model['mean_left_lung'],
                                     model['cov_left_lung'] )
        proba_right_lung = mgaussian( right_lung,
                                      model['mean_right_lung'],
                                      model['cov_right_lung'] )
        proba_liver = mgaussian( liver,
                                 model['mean_liver'],
                                 model['cov_liver'] )
        proba_brain = gaussian( d,
                                model['mean_brain'],
                                model['sigma_brain'] )
        print left_lung, right_lung, liver, proba_left_lung,  proba_right_lung, proba_liver, proba_brain
        scores.append( [proba_left_lung,  proba_right_lung, proba_liver, proba_brain] )
    scores = np.array(scores)
    print scores.mean(axis=0)
    
    average = irtk.imread("/vol/medic02/users/kpk09/gitlab/fetus-detector/body-detector/notebooks/tmp/new_average_heart_center.nii.gz",force_neurological=False)

    irtk.imwrite("average_noheader.nii.gz",irtk.Image(average))
    
    res = irtk.Image( np.zeros(average.shape, dtype='float32') )
    heart = np.array(res.shape,dtype='int')/2

    res2 = irtk.Image( np.zeros(average.shape, dtype='uint8') )
    ball = irtk.Image( morphology.ball(5) )
    for left_lung, right_lung, liver, d in all_coordinates:
        res[heart[0]+left_lung[0],
            heart[1]+left_lung[1],
            heart[2]+left_lung[2]] = 1
        res[heart[0]+right_lung[0],
            heart[1]+right_lung[1],
            heart[2]+right_lung[2]] = 2
        res[heart[0],
            heart[1],
            heart[2]] = 3
        res[heart[0]+liver[0],
            heart[1]+liver[1],
            heart[2]+liver[2]] = 4

        for c,l in [(left_lung,1),
                    (right_lung,2),
                    ([0,0,0],3),
                    (liver,4)]:            
            ball.header['origin'] = res2.ImageToWorld((heart+c)[::-1]).astype('float64')
            ball2 = ball.transform(target=res2)
            res2[ball2>0] = l

    irtk.imwrite('all_centersA.nii.gz',res)
    irtk.imwrite('all_centers.nii.gz',res2)
