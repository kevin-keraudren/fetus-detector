#!/usr/bin/python
"""
 - to handle twisting fetuses: data augmentation with jitter on brain center
 - add narrow band option for training

"""


import irtk
import numpy as np
import os
import cPickle as pickle

import sys
sys.path.append("lib")

from _image_features import ( get_block_means_cpp,
                              get_block_comparisons_cpp,
                              get_grad,
                              get_block_means_cpp_uvw,
                              get_block_comparisons_cpp_uvw,
                              get_grad_comparisons_cpp_uvw,
                              get_grad_uvw,
                              hough_votes,
                              hough_votes_backprojection )
from image_features import integral_image

import scipy.ndimage as nd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

import joblib
from joblib import Parallel, delayed

import argparse

from libdetector import *

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( '--img', type=str, nargs='+' )
parser.add_argument( '--seg', type=str, nargs='+' )
parser.add_argument( '--output_folder', type=str )
parser.add_argument( '--n_jobs', type=int, default=20 )
parser.add_argument( '--n_samples', type=int, default=1000 )
parser.add_argument( '--n_tests', type=int, default=500 )
parser.add_argument( '--o_size', type=int, default=10 )
parser.add_argument( '--d_size', type=int, default=15 )
parser.add_argument( '--n_estimators', type=int, default=100 )
parser.add_argument( '--max_depth', type=int, default=None )
parser.add_argument( '--max_features', type=float, default=0.1 )
parser.add_argument( '--factor_background', type=float, default=2.0 )
parser.add_argument( '--heart_only', action="store_true", default=False )
parser.add_argument( '--not_centered', action="store_true", default=False )
parser.add_argument( '--brain_jitter', type=int, default=10 )
parser.add_argument( '--lung_jitter', type=int, default=5 )
parser.add_argument( '--heart_jitter', type=int, default=5 )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--narrow_band', action="store_true", default=False )
parser.add_argument( '--exclude', type=str, default=None )
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

if args.exclude is not None:
    args.img = filter( lambda x: args.exclude not in x, args.img )
    args.seg = filter( lambda x: args.exclude not in x, args.seg )
     
if args.debug and len(args.img) > 10:
    args.img = args.img[:10]
    args.seg = args.seg[:10]

pickle.dump( args, open( args.output_folder + '/args.pk', 'wb' ) )

nb_labels = 4

offsets1 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')
sizes1 = np.random.randint( 0, args.d_size+1, size=(args.n_tests,1) ).astype('int32')             
offsets2 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')
sizes2 = np.random.randint( 0, args.d_size+1, size=(args.n_tests,1) ).astype('int32')
offsets3 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')
sizes3 = np.random.randint( 0, args.d_size+1, size=(args.n_tests,1) ).astype('int32')             
offsets4 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')
sizes4 = np.random.randint( 0, args.d_size+1, size=(args.n_tests,1) ).astype('int32') 
offsets5 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')
offsets6 = np.random.randint( -args.o_size, args.o_size+1, size=(args.n_tests,3) ).astype('int32')

# we use only squares for rotation invariance
sizes1 = np.tile(sizes1,(1,3))
sizes2 = np.tile(sizes2,(1,3))
sizes3 = np.tile(sizes3,(1,3))
sizes4 = np.tile(sizes4,(1,3))

np.save( args.output_folder + '/offsets1.npy', offsets1 )
np.save( args.output_folder + '/sizes1.npy', sizes1 )
np.save( args.output_folder + '/offsets2.npy', offsets2 )
np.save( args.output_folder + '/sizes2.npy', sizes2 )
np.save( args.output_folder + '/offsets3.npy', offsets3 )
np.save( args.output_folder + '/sizes3.npy', sizes3 )
np.save( args.output_folder + '/offsets4.npy', offsets4 )
np.save( args.output_folder + '/sizes4.npy', sizes4 )
np.save( args.output_folder + '/offsets5.npy', offsets5 )
np.save( args.output_folder + '/offsets6.npy', offsets6 )

def get_training_data_classification( img, seg, label_name ):
    seg = irtk.imread( seg, dtype='int32', force_neurological=True )
    img = irtk.imread( img, dtype='int32', force_neurological=True )

    #u0,v0,w0 = get_orientation_training(seg)
    brain_center, heart_center, left_lung, right_lung = get_centers(seg)
    
    grad = irtk.Image(nd.gaussian_gradient_magnitude( img, 0.5 ),
                      img.get_header())

    blurred_img = nd.gaussian_filter(img,0.5)
    gradZ = nd.sobel( blurred_img, axis=0 ).astype('float32')
    gradY = nd.sobel( blurred_img, axis=1 ).astype('float32')
    gradX = nd.sobel( blurred_img, axis=2 ).astype('float32')

    new_seg = irtk.zeros( seg.get_header() )
    if label_name == "lungs":
        new_seg[seg==3] = 1 # lung 1
        new_seg[seg==4] = 1 # lung 2
    elif label_name == "heart":
        new_seg[seg==5] = 1 # heart
    elif label_name == "liver":
        new_seg[seg==8] = 1 # liver
    seg = new_seg

    sat = integral_image(img)
    sat_grad = integral_image(grad)

    m = np.zeros(img.shape, dtype='uint8')
    m[brain_center[0],
      brain_center[1],
      brain_center[2]] = 1

    narrow_band = nd.distance_transform_edt(np.logical_not(m))
    if args.narrow_band:
        narrow_band[narrow_band<30] = 0
        narrow_band[narrow_band>120] = 0
    narrow_band[img==0] = 0
    
    X = []
    Y = []
    for l in range(2):
        coords = np.argwhere(np.logical_and(narrow_band>0,seg==l))

        if l==0:
            coords = coords[np.random.randint( 0,
                                               coords.shape[0],
                                               int(args.factor_background*args.n_samples))].astype('int32')
        else:
            coords = coords[np.random.randint( 0,
                                               coords.shape[0],
                                               args.n_samples)].astype('int32')

        if args.not_centered:
            x1 = get_block_comparisons_cpp( sat, coords,
                                            offsets1, sizes1,
                                            offsets2, sizes2,
                                            n_jobs=args.n_jobs )
            x2 = get_block_comparisons_cpp( sat_grad, coords,
                                            offsets3, sizes3,
                                            offsets4, sizes4,
                                            n_jobs=args.n_jobs )
            x_grad1 = get_grad( coords.astype('int32'), offsets5.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'))
            x_grad2 = get_grad( coords.astype('int32'), offsets6.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'))
        else:

            if l == 0:
                u,v,w = get_random_orientation(coords.shape[0])
            else:
                # u = np.tile(u0,(coords.shape[0],1))
                # v = np.tile(v0,(coords.shape[0],1))
                # w = np.tile(w0,(coords.shape[0],1))
                u,v,w = get_orientation_training_jitter( brain_center,
                                                         heart_center,
                                                         left_lung,
                                                         right_lung,
                                                         coords.shape[0],
                                                         brain_jitter=args.brain_jitter,
                                                         heart_jitter=args.heart_jitter,
                                                         lung_jitter=args.lung_jitter )

            x1 = get_block_comparisons_cpp_uvw( sat, coords,
                                                w,v,u,
                                                offsets1, sizes1,
                                                offsets2, sizes2,
                                                n_jobs=args.n_jobs )
            x2 = get_block_comparisons_cpp_uvw( sat_grad, coords,
                                                 w,v,u,
                                                offsets3, sizes3,
                                                offsets4, sizes4,
                                                n_jobs=args.n_jobs )
            x3 = get_grad_comparisons_cpp_uvw( gradZ, gradY, gradX,
                                               coords,
                                               offsets5, offsets6,
                                               w,v,u,
                                               n_jobs=args.n_jobs )
            # x_grad1 = get_grad_uvw( coords.astype('int32'), offsets5.astype('int32'),
            #                         gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
            #                         w.astype('float32'), v.astype('float32'), u.astype('float32') )

            # x_grad2 = get_grad_uvw( coords.astype('int32'), offsets6.astype('int32'),
            #                         gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
            #                         w.astype('float32'), v.astype('float32'), u.astype('float32') )

        R = np.linalg.norm( coords - brain_center, axis=1 )
        #x = np.concatenate( ( x1, x2, x_grad1 > x_grad2, R[...,np.newaxis] ), axis=1 )
        x = np.concatenate( ( x1, x2, x3, R[...,np.newaxis] ), axis=1 )
        
        y = seg[coords[:,0],
                coords[:,1],
                coords[:,2]]

        X.extend(x)
        Y.extend(y)

    return (X,Y)

def get_training_data_regression( img, seg ):
    seg = irtk.imread( seg, dtype='int32', force_neurological=True )
    img = irtk.imread( img, dtype='int32', force_neurological=True )

    #u0,v0,w0 = get_orientation_training(seg)
    brain_center, heart_center, left_lung, right_lung = get_centers(seg)
    
    grad = irtk.Image(nd.gaussian_gradient_magnitude( img, 0.5 ),
                      img.get_header())

    blurred_img = nd.gaussian_filter(img,0.5)
    gradZ = nd.sobel( blurred_img, axis=0 ).astype('float32')
    gradY = nd.sobel( blurred_img, axis=1 ).astype('float32')
    gradX = nd.sobel( blurred_img, axis=2 ).astype('float32')

    new_seg = irtk.zeros( seg.get_header() )
    new_seg[seg==3] = 1 # lung 1
    new_seg[seg==4] = 1 # lung 2
    new_seg[seg==5] = 2 # heart
    new_seg[seg==8] = 3 # liver
    seg = new_seg

    center1 =  np.array(nd.center_of_mass( (seg == 1).view(np.ndarray) ),
                        dtype='float32')
    center2 =  np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                        dtype='float32')
    center3 =  np.array(nd.center_of_mass( (seg == 3).view(np.ndarray) ),
                        dtype='float32')
    
    sat = integral_image(img)
    sat_grad = integral_image(grad)

    m = np.zeros(img.shape, dtype='uint8')
    m[brain_center[0],
      brain_center[1],
      brain_center[2]] = 1
    
    X = []
    Y = []
    for l in range(1,nb_labels):
        coords = np.argwhere(seg==l)

        coords = coords[np.random.randint( 0,
                                           coords.shape[0],
                                           args.n_samples)].astype('int32')

        if args.not_centered:
            x1 = get_block_comparisons_cpp( sat, coords,
                                            offsets1, sizes1,
                                            offsets2, sizes2,n_jobs=args.n_jobs )
            x2 = get_block_comparisons_cpp( sat_grad, coords,
                                            offsets3, sizes3,
                                            offsets4, sizes4,
                                            n_jobs=args.n_jobs )
            x_grad1 = get_grad( coords.astype('int32'), offsets5.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
            x_grad2 = get_grad( coords.astype('int32'), offsets6.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
        else:
            # u = np.tile(u0,(coords.shape[0],1))
            # v = np.tile(v0,(coords.shape[0],1))
            # w = np.tile(w0,(coords.shape[0],1))
            u,v,w = get_orientation_training_jitter( brain_center,
                                                     heart_center,
                                                     left_lung,
                                                     right_lung,
                                                     coords.shape[0],
                                                     brain_jitter=args.brain_jitter,
                                                     heart_jitter=args.heart_jitter,
                                                     lung_jitter=args.lung_jitter )

            x1 = get_block_comparisons_cpp_uvw( sat, coords,
                                                w,v,u,
                                                offsets1, sizes1,
                                                offsets2, sizes2,n_jobs=args.n_jobs )
            x2 = get_block_comparisons_cpp_uvw( sat_grad, coords,
                                                w,v,u,
                                                offsets3, sizes3,
                                                offsets4, sizes4,
                                                n_jobs=args.n_jobs )
            x3 = get_grad_comparisons_cpp_uvw( gradZ, gradY, gradX,
                                               coords,
                                               offsets5, offsets6,
                                               w,v,u,
                                               n_jobs=args.n_jobs )
            # x_grad1 = get_grad_uvw( coords.astype('int32'), offsets5.astype('int32'),
            #                         gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
            #                         w.astype('float32'), v.astype('float32'), u.astype('float32') )

            # x_grad2 = get_grad_uvw( coords.astype('int32'), offsets6.astype('int32'),
            #                         gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
            #                         w.astype('float32'), v.astype('float32'), u.astype('float32') )
             
        #x = np.concatenate( ( x1, x2, x_grad1 > x_grad2 ), axis=1 )
        x = np.concatenate( ( x1, x2, x3 ), axis=1 )

        if l == 1:
            y = center1[np.newaxis,...] - coords.astype('float32')
            if not args.not_centered:
                y = np.concatenate( ( (y*w).sum(axis=1)[...,np.newaxis],
                                      (y*v).sum(axis=1)[...,np.newaxis],
                                      (y*u).sum(axis=1)[...,np.newaxis] ),
                                    axis=1 )
        elif l == 2:
            y = center2[np.newaxis,...] - coords.astype('float32')
            if not args.not_centered:
                y = np.concatenate( ( (y*w).sum(axis=1)[...,np.newaxis],
                                      (y*v).sum(axis=1)[...,np.newaxis],
                                      (y*u).sum(axis=1)[...,np.newaxis] ),
                                    axis=1 )
        else:
            y = center3[np.newaxis,...] - coords.astype('float32')
            if not args.not_centered:
                y = np.concatenate( ( (y*w).sum(axis=1)[...,np.newaxis],
                                      (y*v).sum(axis=1)[...,np.newaxis],
                                      (y*u).sum(axis=1)[...,np.newaxis] ),
                                    axis=1 )

        X.append(x)
        Y.append(y)
            
    return X,Y


clf_lungs = RandomForestClassifier( n_estimators=args.n_estimators,
                                    oob_score=True,
                                    #max_depth=args.max_depth,
                                    #max_features=args.max_features,
                                    n_jobs=args.n_jobs )#,min_samples_leaf=10)
clf_heart = RandomForestClassifier( n_estimators=args.n_estimators,
                                    oob_score=True,
                                    #max_depth=args.max_depth,
                                    #max_features=args.max_features,
                                    n_jobs=args.n_jobs )#,min_samples_leaf=10)
clf_liver = RandomForestClassifier( n_estimators=args.n_estimators,
                                    oob_score=True,
                                    #max_depth=args.max_depth,
                                    #max_features=args.max_features,
                                    n_jobs=args.n_jobs )#,min_samples_leaf=10)

# Classification
if not args.heart_only:
    XY = Parallel(n_jobs=args.n_jobs)(delayed(get_training_data_classification)(img,seg,"lungs")
                             for img,seg in zip(args.img,args.seg) )

    X_train = []
    Y_train = []
    for x,y in XY:
        X_train.extend(x)
        Y_train.extend(y)

    print "train:",len(X_train),len(Y_train)

    clf_lungs.fit(X_train,Y_train)
    print "OOB clf_lungs:",clf_lungs.oob_score_
    np.save( args.output_folder + "/feature_importance_clf_lungs.npy",
             clf_lungs.feature_importances_ )
    joblib.dump( clf_lungs, args.output_folder+'/clf_lungs' )

XY = Parallel(n_jobs=args.n_jobs)(delayed(get_training_data_classification)(img,seg,"heart")
                         for img,seg in zip(args.img,args.seg) )

X_train = []
Y_train = []
for x,y in XY:
    X_train.extend(x)
    Y_train.extend(y)
    
print "train:",len(X_train),len(Y_train)

clf_heart.fit(X_train,Y_train)
print "OOB clf_heart:",clf_heart.oob_score_
np.save( args.output_folder + "/feature_importance_clf_heart.npy",
         clf_heart.feature_importances_ )
joblib.dump( clf_heart, args.output_folder+'/clf_heart' )

if not args.heart_only:
    XY = Parallel(n_jobs=args.n_jobs)(delayed(get_training_data_classification)(img,seg,"liver")
                                          for img,seg in zip(args.img,args.seg) )

    X_train = []
    Y_train = []
    for x,y in XY:
        X_train.extend(x)
        Y_train.extend(y)

    print "train:",len(X_train),len(Y_train)

    clf_liver.fit(X_train,Y_train)
    print "OOB clf_liver:",clf_liver.oob_score_
    np.save( args.output_folder + "/feature_importance_clf_liver.npy",
             clf_liver.feature_importances_ )
    joblib.dump( clf_liver, args.output_folder+'/clf_liver' )

# Regression
XY = Parallel(n_jobs=args.n_jobs)(delayed(get_training_data_regression)(img,seg)
                                      for img,seg in zip(args.img,args.seg) )

X_lungs = []
Y_lungs = []
X_heart = []
Y_heart = []
X_liver = []
Y_liver = []
for (xlu,xh,xli),(ylu,yh,yli) in XY:
    X_lungs.extend(xlu)
    Y_lungs.extend(ylu)
    X_heart.extend(xh)
    Y_heart.extend(yh)
    X_liver.extend(xli)
    Y_liver.extend(yli)

print "train:",len(X_lungs),len(Y_lungs),len(X_heart),len(Y_heart),len(X_liver),len(Y_liver)

reg_lungs = RandomForestRegressor( n_estimators=args.n_estimators,
                                oob_score=True,
                                n_jobs=args.n_jobs )#,min_samples_leaf=10)
reg_heart = RandomForestRegressor( n_estimators=args.n_estimators,
                                      oob_score=True,
                                      n_jobs=args.n_jobs )#,min_samples_leaf=10)
reg_liver = RandomForestRegressor( n_estimators=args.n_estimators,
                                      oob_score=True,
                                      n_jobs=args.n_jobs )#,min_samples_leaf=10)

if not args.heart_only:
    reg_lungs.fit(X_lungs,Y_lungs)
    print "OOB reg_lungs:",reg_lungs.oob_score_
    np.save( args.output_folder + "/feature_importance_reg_lungs.npy",
             reg_lungs.feature_importances_ )
    joblib.dump( reg_lungs, args.output_folder+'/reg_lungs' )

reg_heart.fit(X_heart,Y_heart)
print "OOB reg_heart:",reg_heart.oob_score_
np.save( args.output_folder + "/feature_importance_reg_heart.npy",
         reg_heart.feature_importances_ )
joblib.dump( reg_heart, args.output_folder+'/reg_heart' )

if not args.heart_only:
    reg_liver.fit(X_liver,Y_liver)
    print "OOB reg_liver:",reg_liver.oob_score_
    np.save( args.output_folder + "/feature_importance_reg_liver.npy",
             reg_liver.feature_importances_ )
    joblib.dump( reg_liver, args.output_folder+'/reg_liver' )


