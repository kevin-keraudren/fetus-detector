#!/usr/bin/python

import irtk
import numpy as np
import os

import sys
sys.path.append("lib")

from _image_features import ( get_block_means_cpp,
                              get_block_comparisons_cpp,
                              get_grad,
                              get_block_means_cpp_uvw,
                              get_block_comparisons_cpp_uvw,
                              get_grad_uvw,
                              hough_votes,
                              hough_votes_backprojection )
from image_features import integral_image

from glob import glob

import scipy.ndimage as nd
from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel, delayed

from libdetector import *

myfolder = "/vol/bitbucket/kpk09/detector/data_resampled"
n_samples = 1000
n_tests = 500

all_files = glob(myfolder+"/*_img.nii.gz")

np.random.shuffle(all_files)

mygrid = []
for o_size in [1,5,10,15,20,25,30]:
    for d_size in [1,5,10,15,20,25,30]:
        mygrid.append( (o_size,d_size) )

def run( o_size, d_size):
    offsets1 = np.random.randint( -o_size, o_size+1, size=(n_tests,3) ).astype('int32')
    sizes1 = np.random.randint( 0, d_size+1, size=(n_tests,1) ).astype('int32')             
    offsets2 = np.random.randint( -o_size, o_size+1, size=(n_tests,3) ).astype('int32')
    sizes2 = np.random.randint( 0, d_size+1, size=(n_tests,1) ).astype('int32')
    offsets3 = np.random.randint( -o_size, o_size+1, size=(n_tests,3) ).astype('int32')
    sizes3 = np.random.randint( 0, d_size+1, size=(n_tests,1) ).astype('int32')             
    offsets4 = np.random.randint( -o_size, o_size+1, size=(n_tests,3) ).astype('int32')
    sizes4 = np.random.randint( 0, d_size+1, size=(n_tests,1) ).astype('int32') 
    offsets5 = np.random.randint( -o_size, o_size+1, size=(n_tests/3,3) ).astype('int32')
    offsets6 = np.random.randint( -o_size, o_size+1, size=(n_tests/3,3) ).astype('int32')

    # we use only squares for rotation invariance
    sizes1 = np.tile(sizes1,(1,3))
    sizes2 = np.tile(sizes2,(1,3))
    sizes3 = np.tile(sizes3,(1,3))
    sizes4 = np.tile(sizes4,(1,3))
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    n = 0
    for f in all_files:
        n += 1
        patient_id = os.path.basename(f)[:-len("_img.nii.gz")]
        img = irtk.imread( f, dtype='int32', force_neurological=True )
        seg = irtk.imread( myfolder + "/" + patient_id + "_seg.nii.gz", dtype='int32', force_neurological=True )

        brain_center, heart_center, left_lung, right_lung = get_centers(seg)
    
        grad = irtk.Image(nd.gaussian_gradient_magnitude( img, 0.5 ),
                          img.get_header())

        blurred_img = nd.gaussian_filter(img,0.5)
        gradZ = nd.sobel( blurred_img, axis=0 )
        gradY = nd.sobel( blurred_img, axis=1 )
        gradX = nd.sobel( blurred_img, axis=2 )

        new_seg = irtk.zeros( seg.get_header() )
        new_seg[seg==5] = 1 # heart
        seg = new_seg

        sat = integral_image(img)
        sat_grad = integral_image(grad)

        m = np.zeros(img.shape, dtype='uint8')
        m[brain_center[0],
          brain_center[1],
          brain_center[2]] = 1

        narrow_band = nd.distance_transform_edt(np.logical_not(m))
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
                                                   n_samples)].astype('int32')
            else:
                coords = coords[np.random.randint( 0,
                                                   coords.shape[0],
                                                   n_samples)].astype('int32')

            if l == 0:
                u,v,w = get_random_orientation(coords.shape[0])
            else:
                u,v,w = get_orientation_training_jitter( brain_center,
                                                         heart_center,
                                                         left_lung,
                                                         right_lung,
                                                         coords.shape[0],
                                                         brain_jitter=10,
                                                         heart_jitter=5,
                                                         lung_jitter=5 )

            x1 = get_block_comparisons_cpp_uvw( sat, coords,
                                                w,v,u,
                                                offsets1, sizes1,
                                                offsets2, sizes2,
                                                n_jobs=10 )
            x2 = get_block_comparisons_cpp_uvw( sat_grad, coords,
                                                 w,v,u,
                                                offsets3, sizes3,
                                                offsets4, sizes4,
                                                n_jobs=10 )
            x_grad1 = get_grad_uvw( coords.astype('int32'), offsets5.astype('int32'),
                                    gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
                                    w.astype('float32'), v.astype('float32'), u.astype('float32') )

            x_grad2 = get_grad_uvw( coords.astype('int32'), offsets6.astype('int32'),
                                    gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32'),
                                    w.astype('float32'), v.astype('float32'), u.astype('float32') )

            x = np.concatenate( ( x1, x2, x_grad1 > x_grad2 ), axis=1 )
            #x = np.concatenate( ( x1, x2 ), axis=1 )
        
            y = seg[coords[:,0],
                    coords[:,1],
                    coords[:,2]]
            
            if n < 30:
                X_train.extend(x)
                Y_train.extend(y)
            else:
                X_test.extend(x)
                Y_test.extend(y)

    print "train:",len(X_train),len(Y_train)
    forest = RandomForestClassifier( n_estimators=100,
                                     oob_score=True,
                                     n_jobs=10 )#,min_samples_leaf=10)

    forest.fit(X_train,Y_train)
    oob_score = forest.oob_score_

    print "test:",len(X_test),len(Y_test)
    test_score = forest.score( X_test, Y_test)

    print [oob_score,test_score,o_size,d_size]
    return [oob_score,test_score,o_size,d_size]

scores = Parallel(n_jobs=5)(delayed(run)(o_size,d_size)
                         for o_size,d_size in mygrid )

scores = np.array(scores)
np.save("size_scores.npy",scores)

