#!/usr/bin/python

import irtk
import numpy as np
import os

import sys
sys.path.insert(1,os.path.dirname(os.path.abspath(__file__))+"/lib")
sys.path.insert(1,os.path.dirname(os.path.abspath(__file__))+"/../commonlib")

from _image_features import ( get_block_means_cpp,
                              get_block_comparisons_cpp,
                              get_grad,
                              get_block_means_cpp_uvw,
                              get_block_comparisons_cpp_uvw,
                              get_grad_comparisons_cpp_uvw,
                              get_grad_uvw,
                              get_grad_comparisons_uvw,
                              hough_votes,
                              hough_votes_backprojection )
from image_features import integral_image
from glob import glob

import scipy.ndimage as nd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import joblib

import cPickle as pickle

import argparse

from libdetector import *
from helpers import *

from skimage.feature import peak_local_max

import gc

def get_orientation2( brain_center,
                      heart_center,
                      coords,
                      brain_jitter=5,
                      heart_jitter=5 ):
    u = ( ( brain_center +
            np.random.randint(-brain_jitter, brain_jitter+1,
                              size=(coords.shape[0],3) )) -
          ( heart_center +
            np.random.randint(-heart_jitter, heart_jitter+1,
                              size=(coords.shape[0],3) )) )
    u /= np.linalg.norm( u, axis=1 )[...,np.newaxis]

    # np.random.rand() returns random floats in the interval [0;1)
    v = 2*np.random.rand( coords.shape[0], 3 ) - 1
    v -= (v*u).sum(axis=1)[...,np.newaxis]*u
    v /= np.linalg.norm( v, axis=1 )[...,np.newaxis]

    w = np.cross( u, v )
    w /= np.linalg.norm(w, axis=1)[...,np.newaxis]

    return ( u.astype("float32"),
             v.astype("float32"),
             w.astype("float32") )

parser = argparse.ArgumentParser(
        description='' )
# required parameters
parser.add_argument( "--input", type=str, required=True, action=PathExists )
parser.add_argument( "--detector", type=str, required=True, action=PathExists )
parser.add_argument( "--shape_model", type=str, required=True, action=PathExists )
parser.add_argument( '--brain_center', type=float, nargs=3, required=True )
parser.add_argument( '--heart_center', type=float, nargs=3, required=True )
# parameters with default values
parser.add_argument( "--output", type=str )
parser.add_argument( '--n_jobs', type=int, default=20 )
parser.add_argument( '--chunk_size', type=int, default=int(3e6) )
parser.add_argument( '--fast', type=float, default=0.0 )
parser.add_argument( '--narrow_band', action="store_true", default=False )
parser.add_argument( '--brain_jitter', type=int, default=10 )
parser.add_argument( '--heart_jitter', type=int, default=10 )
parser.add_argument( '--theta', type=int, default=90 )
parser.add_argument( '--padding', type=int, default=10 )
parser.add_argument( '-l', type=float, default=0.5 )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--not_centered', action="store_true", default=False )
parser.add_argument( '--aligned', action="store_true", default=False )
parser.add_argument( '--random', action="store_true", default=False )
parser.add_argument( '--selective', action="store_true", default=False )
parser.add_argument( '--back_proj', action="store_true", default=False )
parser.add_argument( '--score', action="store_true", default=False )
parser.add_argument( '--verbose', action="store_true", default=False )
args = parser.parse_args()

if not args.score:
    if args.output is None:
        sys.stderr.write( "missing --output" )
        exit(1)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

nb_labels = 5

training_args = pickle.load( open( args.detector + '/args.pk' ) )

offsets1 = np.load( args.detector + '/offsets1.npy' )
sizes1 = np.load( args.detector + '/sizes1.npy' )
offsets2 = np.load( args.detector + '/offsets2.npy' )
sizes2 = np.load( args.detector + '/sizes2.npy' )
offsets3 = np.load( args.detector + '/offsets3.npy' )
sizes3 = np.load( args.detector + '/sizes3.npy' )
offsets4 = np.load( args.detector + '/offsets4.npy' )
sizes4 = np.load( args.detector + '/sizes4.npy' )
offsets5 = np.load( args.detector + '/offsets5.npy' )
offsets6 = np.load( args.detector + '/offsets6.npy' )

clf = joblib.load( args.detector + '/clf' )
reg_heart = joblib.load( args.detector + '/reg_heart' )
reg_left_lung = joblib.load( args.detector + '/reg_left_lung' )
reg_right_lung = joblib.load( args.detector + '/reg_right_lung' )
reg_liver = joblib.load( args.detector + '/reg_liver' )

clf.set_params(n_jobs=args.n_jobs)
reg_heart.set_params(n_jobs=args.n_jobs)
reg_left_lung.set_params(n_jobs=args.n_jobs)
reg_right_lung.set_params(n_jobs=args.n_jobs)
reg_liver.set_params(n_jobs=args.n_jobs)

if args.verbose:
    print "forests loaded"

img = irtk.imread( args.input, dtype='float32', force_neurological=True )

grad = irtk.Image(nd.gaussian_gradient_magnitude( img, 0.5 ),
              img.get_header())

sat = integral_image(img)
sat_grad = integral_image(grad)

blurred_img = nd.gaussian_filter(img,0.5)
gradZ = nd.sobel( blurred_img, axis=0 ).astype('float32')
gradY = nd.sobel( blurred_img, axis=1 ).astype('float32')
gradX = nd.sobel( blurred_img, axis=2 ).astype('float32')

if not args.score:
    irtk.imwrite(args.output + "/img.nii.gz", img)
    irtk.imwrite(args.output + "/grad.nii.gz", grad)

del blurred_img
del grad

if args.verbose:
    print "preprocessing done"

args.brain_center = np.array(args.brain_center,dtype='float32') 
args.heart_center = np.array(args.heart_center,dtype='float32') 
    
narrow_band = get_narrow_band( img.shape, args.heart_center, r_max=50 )
narrow_band[img==0] = 0

narrow_band = irtk.Image(narrow_band,img.get_header())

if args.fast:
    narrow_band_downsampled = narrow_band.resample(img.header['pixelSize']*args.fast)
    img_downsampled = img.resample(img.header['pixelSize']*args.fast)
    #coords2 = np.argwhere(img_downsampled>0).astype('int32')
    coords2 = np.argwhere(narrow_band_downsampled>0).astype('int32')
    coords = img.WorldToImage(img_downsampled.ImageToWorld(coords2[:,::-1]))[:,::-1].astype('int32')
    header = img_downsampled.get_header()
else:
    #coords = np.argwhere(img>0).astype('int32')
    coords = np.argwhere(narrow_band>0).astype('int32')
    header = img.get_header()

if args.chunk_size > len(coords)/2:
    args.chunk_size = len(coords)/2
    if args.verbose:
        print "chunk size too large, setting it to", args.chunk_size
    
header['dim'][3] = nb_labels
proba = irtk.zeros(header,dtype='float32')

if args.aligned:
    u = np.zeros((coords.shape[0],3),dtype='float32')
    u[:,2] = 1 # heart to brain
    v0 = np.zeros((coords.shape[0],3),dtype='float32')
    v0[:,1] = 1 # left to right
    # w0 = np.zeros((coords.shape[0],3),dtype='float32')
    # w0[:,0] = -1 # np.cross(u,v)
    w0 = np.cross(u,v0)
    args.theta = 360
elif args.random:
    # If the location of the brain is unknown,
    # we can try random orientations
    u,v0,w0 =  get_random_orientation(coords.shape[0])
else:
    u,v0,w0 = get_orientation2( args.brain_center,
                                args.heart_center,
                                coords )
    
best_proba = irtk.zeros(header,dtype='float32')
if args.fast:
    best_proba[ 0,
                coords2[:,0],
                coords2[:,1],
                coords2[:,2]] = np.finfo('float32').max
else:
    best_proba[ 0,
                coords[:,0],
                coords[:,1],
                coords[:,2]] = np.finfo('float32').max
    
best_v = v0.copy()
best_w = w0.copy()

for theta in xrange(0,360,args.theta):
    if args.verbose:
        print "theta =",theta
    v = np.cos(float(theta)/180*np.pi)*v0 + np.sin(float(theta)/180*np.pi)*w0
    w = np.cos(float(theta+90)/180*np.pi)*v0 + np.sin(float(theta+90)/180*np.pi)*w0
    for i in xrange(0,coords.shape[0],args.chunk_size):
        j = min(i+args.chunk_size,coords.shape[0])
        # pre-allocating memory
        x = np.zeros( (j-i,offsets1.shape[0]+offsets3.shape[0]+offsets5.shape[0]+2), dtype='float32' )
        if args.not_centered:
             x1 = get_block_comparisons_cpp( sat, np.ascontiguousarray(coords[i:j]),
                                             offsets1, sizes1,
                                             offsets2, sizes2,n_jobs=args.n_jobs )
             x2 = get_block_comparisons_cpp( sat_grad, np.ascontiguousarray(coords[i:j]),
                                             offsets3, sizes3,
                                             offsets4, sizes4, n_jobs=args.n_jobs)
             x_grad1 = get_grad( np.ascontiguousarray(coords[i:j].astype('int32')),
                                 offsets5.astype('int32'),
                                 gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
             x_grad2 = get_grad( np.ascontiguousarray(coords[i:j].astype('int32')),
                                 offsets6.astype('int32'),
                                 gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
        else:
            x[:,:offsets1.shape[0]] = get_block_comparisons_cpp_uvw( sat,
                                                np.ascontiguousarray(coords[i:j]),
                                                np.ascontiguousarray(w[i:j]),
                                                np.ascontiguousarray(v[i:j]),
                                                np.ascontiguousarray(u[i:j]),
                                                offsets1, sizes1,
                                                offsets2, sizes2, n_jobs=args.n_jobs )
            feature_offset = offsets1.shape[0]
            
            x[:,feature_offset:feature_offset+offsets3.shape[0]] = get_block_comparisons_cpp_uvw( sat_grad,
                                                np.ascontiguousarray(coords[i:j]),
                                                np.ascontiguousarray(w[i:j]),
                                                np.ascontiguousarray(v[i:j]),
                                                np.ascontiguousarray(u[i:j]),
                                                offsets3, sizes3,
                                                offsets4, sizes4, n_jobs=args.n_jobs)
            feature_offset += offsets3.shape[0]

            x[:,feature_offset:feature_offset+offsets5.shape[0]] = get_grad_comparisons_cpp_uvw(
                gradZ, gradY, gradX,
                np.ascontiguousarray(coords[i:j].astype('int32')),
                offsets5,
                offsets6,
                np.ascontiguousarray(w[i:j].astype('float32')),
                np.ascontiguousarray(v[i:j].astype('float32')),
                np.ascontiguousarray(u[i:j].astype('float32')) )

            feature_offset += offsets5.shape[0]
            
        x[:,feature_offset] = np.sum( (coords[i:j] - args.heart_center)*u[i:j], axis=1 )
        x[:,feature_offset+1] = np.linalg.norm( coords[i:j] - args.heart_center, axis=1 )

        pr = clf.predict_proba(x)

        if args.fast:
            selection = pr[:,0] < best_proba[0,
                                             coords2[i:j,0],
                                             coords2[i:j,1],
                                             coords2[i:j,2]]
            best_proba[ 0,
                        coords2[i:j,0][selection],
                        coords2[i:j,1][selection],
                        coords2[i:j,2][selection] ] = pr[:,0][selection]
            selection_vw = pr[:,1:].max(axis=1) > best_proba[1:,
                                                             coords2[i:j,0],
                                                             coords2[i:j,1],
                                                             coords2[i:j,2]].max(axis=0)
            best_v[i:j][selection_vw] = v[i:j][selection_vw]
            best_w[i:j][selection_vw] = w[i:j][selection_vw]
            for dim in xrange(1,nb_labels):
                selection = pr[:,dim] > best_proba[dim,
                                                   coords2[i:j,0],
                                                   coords2[i:j,1],
                                                   coords2[i:j,2]]
                best_proba[ dim,
                            coords2[i:j,0][selection],
                            coords2[i:j,1][selection],
                            coords2[i:j,2][selection] ] = pr[:,dim][selection] 
        else:
            selection = pr[:,0] < best_proba[0,
                                             coords[i:j,0],
                                             coords[i:j,1],
                                             coords[i:j,2]]
            best_proba[ 0,
                        coords[i:j,0][selection],
                        coords[i:j,1][selection],
                        coords[i:j,2][selection] ] = pr[:,0][selection]
            selection_vw = pr[:,1:].max(axis=1) > best_proba[1:,
                                                             coords[i:j,0],
                                                             coords[i:j,1],
                                                             coords[i:j,2]].max(axis=0)
            best_v[i:j][selection_vw] = v[i:j][selection_vw]
            best_w[i:j][selection_vw] = w[i:j][selection_vw]
            for dim in xrange(1,nb_labels):
                selection = pr[:,dim] > best_proba[dim,
                                                   coords[i:j,0],
                                                   coords[i:j,1],
                                                   coords[i:j,2]]
                best_proba[ dim,
                            coords[i:j,0][selection],
                            coords[i:j,1][selection],
                            coords[i:j,2][selection] ] = pr[:,dim][selection]

del x
_ = gc.collect()

v = best_v
w = best_w
proba = best_proba
        
if args.fast:
    header = img.get_header()
    header['dim'][3] = nb_labels
    proba = proba.transform(target=header)

if not args.score:
    irtk.imwrite( args.output + "/proba.nii.gz", proba )

seg = np.argmax( proba, axis=0)
seg = irtk.Image( seg,
                  img.get_header() ).astype('uint8')

if not args.score:
    irtk.imwrite( args.output + "/seg.nii.gz", seg )

# Regression
header = img.get_header()
header['dim'][3] = 3
offsets_heart = irtk.ones(header,dtype='float32') * np.finfo('float32').max
offsets_left_lung = irtk.ones(header,dtype='float32') * np.finfo('float32').max
offsets_right_lung = irtk.ones(header,dtype='float32') * np.finfo('float32').max
offsets_liver = irtk.ones(header,dtype='float32') * np.finfo('float32').max
for l, forest, offsets in zip( [1,2,3,4],#range(1,nb_labels),
                               [reg_left_lung,reg_right_lung,reg_heart,reg_liver],
                               [offsets_left_lung,offsets_right_lung,offsets_heart,offsets_liver] ):  

    selection = seg[coords[:,0],
                    coords[:,1],
                    coords[:,2]] == l
    
    coords_reg = coords[selection].astype('int32').copy()
    u2 = u[selection].copy()
    v2 = v[selection].copy()
    w2 = w[selection].copy()

    for i in xrange(0,coords_reg.shape[0],args.chunk_size):
        j = min(i+args.chunk_size,coords_reg.shape[0])
        # pre-allocating memory
        x = np.zeros( (j-i,offsets1.shape[0]+offsets3.shape[0]+offsets5.shape[0]), dtype='float32' )
        if args.not_centered:
            x1 = get_block_comparisons_cpp( sat, coords_reg[i:j],
                                            offsets1, sizes1,
                                            offsets2, sizes2,
                                            n_jobs=args.n_jobs)
            x2 = get_block_comparisons_cpp( sat_grad, coords_reg[i:j],
                                            offsets3, sizes3,
                                            offsets4, sizes4,
                                            n_jobs=args.n_jobs)

            x_grad1 = get_grad( coords_reg[i:j].astype('int32'), offsets5.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
            x_grad2 = get_grad( coords_reg[i:j].astype('int32'), offsets6.astype('int32'),
                                gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
        else:
            x[:,:offsets1.shape[0]] = get_block_comparisons_cpp_uvw( sat, coords_reg[i:j],
                                                w2[i:j],v2[i:j],u2[i:j],
                                                offsets1, sizes1,
                                                offsets2, sizes2,
                                                n_jobs=args.n_jobs)
            feature_offset = offsets1.shape[0]
            
            x[:,feature_offset:feature_offset+offsets3.shape[0]] = get_block_comparisons_cpp_uvw( sat_grad, coords_reg[i:j],
                                                w2[i:j],v2[i:j],u2[i:j],
                                                offsets3, sizes3,
                                                offsets4, sizes4,
                                                n_jobs=args.n_jobs)
            feature_offset += offsets3.shape[0]

            x[:,feature_offset:feature_offset+offsets5.shape[0]] = get_grad_comparisons_cpp_uvw(
                gradZ, gradY, gradX,
                coords_reg[i:j].astype('int32'),
                offsets5,
                offsets6,
                w2[i:j].astype('float32'), v2[i:j].astype('float32'), u2[i:j].astype('float32') )
        
        y = forest.predict(x)

        if not args.not_centered:
            y = y[:,0][...,np.newaxis]*w2[i:j] + y[:,1][...,np.newaxis]*v2[i:j] + y[:,2][...,np.newaxis]*u2[i:j]

        for dim in xrange(3):
            offsets[dim,
                    coords_reg[i:j,0],
                    coords_reg[i:j,1],
                    coords_reg[i:j,2]] = y[:,dim]

del x
_ = gc.collect()
            
if not args.score:            
    irtk.imwrite( args.output + "/offsets_left_lung.nii.gz",
                  offsets_left_lung )
    irtk.imwrite( args.output + "/offsets_right_lung.nii.gz",
                  offsets_right_lung )
    irtk.imwrite( args.output + "/offsets_heart.nii.gz",
                  offsets_heart )
    irtk.imwrite( args.output + "/offsets_liver.nii.gz",
                  offsets_liver )

hough_header = img.get_header()
hough_header['dim'][:3] += 2*args.padding

votes_left_lung = hough_votes(np.ascontiguousarray(np.round(offsets_left_lung).astype('int32')),
                            proba[1], padding=args.padding )
votes_right_lung = hough_votes(np.ascontiguousarray(np.round(offsets_right_lung).astype('int32')),
                            proba[2], padding=args.padding )
if not args.score:
    irtk.imwrite(args.output + "/votes_left_lung.nii.gz",
                 irtk.Image(votes_left_lung,hough_header))
    irtk.imwrite(args.output + "/votes_right_lung.nii.gz",
                 irtk.Image(votes_right_lung,hough_header))    

votes_heart = hough_votes(np.ascontiguousarray(np.round(offsets_heart).astype('int32')),
                            proba[3], padding=args.padding )
if not args.score:
    irtk.imwrite(args.output + "/votes_heart.nii.gz",
                 irtk.Image(votes_heart,hough_header))

votes_liver = hough_votes(np.ascontiguousarray(np.round(offsets_liver).astype('int32')),
                            proba[4], padding=args.padding )
if not args.score:
    irtk.imwrite(args.output + "/votes_liver.nii.gz",
                 irtk.Image(votes_liver,hough_header)) 

# # mask using narrow_band
# if args.padding:
#     # update the narrow band without forgetting to
#     # translate the heart_center
#     narrow_band = get_narrow_band( votes_heart.shape, args.heart_center+args.padding, r_max=50 )
    
# votes_left_lung[narrow_band==0] = 0
# votes_right_lung[narrow_band==0] = 0
# votes_heart[narrow_band==0] = 0
# votes_liver[narrow_band==0] = 0
    
votes_left_lung, scale_left_lung = rescale( nd.gaussian_filter(votes_left_lung, 2.0), dtype='float32' )
votes_right_lung, scale_right_lung = rescale( nd.gaussian_filter(votes_right_lung, 2.0), dtype='float32' )
votes_heart, scale_heart = rescale( nd.gaussian_filter(votes_heart, 2.0), dtype='float32' )
votes_liver, scale_liver = rescale( nd.gaussian_filter(votes_liver, 2.0), dtype='float32' )

shape_model = pickle.load( open(args.shape_model, 'rb' ) )
p_heart = np.unravel_index( np.argmax(votes_heart), votes_heart.shape )
p_brain = args.brain_center+args.padding

sys.stderr.write( "Looking for the best candidates\n" )

candidates_left_lung = peak_local_max( votes_left_lung,
                                       min_distance=10,
                                       threshold_rel=0.2,
                                       exclude_border=False,
                                       indices=True,
                                       num_peaks=5 )
candidates_right_lung = peak_local_max( votes_right_lung,
                                        min_distance=10,
                                        threshold_rel=0.2,
                                        exclude_border=False,
                                        indices=True,
                                        num_peaks=5 )
candidates_liver = peak_local_max( votes_liver,
                                   min_distance=10,
                                   threshold_rel=0.2,
                                   exclude_border=False,
                                   indices=True,
                                   num_peaks=5 )

candidates = []
scores = []
for left_lung in candidates_left_lung:
    for right_lung in candidates_right_lung:
        for liver in candidates_liver:
            v_left_lung = votes_left_lung[left_lung[0],left_lung[1],left_lung[2]]
            v_right_lung = votes_right_lung[right_lung[0],right_lung[1],right_lung[2]]
            v_liver = votes_liver[liver[0],liver[1],liver[2]]
            s = v_left_lung + v_right_lung + v_liver
            d = shape_proba( np.array( p_heart, dtype='float', copy=True ),
                             np.array( p_brain, dtype='float', copy=True ),
                             np.array( left_lung, dtype='float', copy=True ),
                             np.array( right_lung, dtype='float', copy=True ),
                             np.array( liver, dtype='float', copy=True ),
                             shape_model,
                             verbose=args.verbose )
            if args.verbose:
                print left_lung, right_lung, liver
                print v_left_lung, v_right_lung, v_liver
                print "S =", s, ";\tD =", d, "\tscore =", args.l*s + (1.0-args.l)*d
            candidates.append( [ (left_lung,right_lung,liver),
                                 (v_left_lung*scale_left_lung,
                                  v_right_lung*scale_right_lung,
                                  v_liver*scale_liver,
                                  d)])
            scores.append( args.l*s + (1.0-args.l)*d )

if args.verbose:
    print scores

if len(scores) == 0:
    print 0, 0, 0, 0
    exit(0)

scores = np.array(scores)
score = scores.max()
sys.stderr.write( str(score) + "\n" )
best_candidates = np.argmax(scores)
(p_left_lung, p_right_lung, p_liver), (pr_left_lung, pr_right_lung, pr_liver, d) = candidates[best_candidates]

if not args.score:
    # p_left_lung = np.unravel_index( np.argmax(votes_left_lung), img.shape )
    # p_right_lung = np.unravel_index( np.argmax(votes_right_lung), img.shape )
    # p_liver = np.unravel_index( np.argmax(votes_liver), img.shape )
    # p_heart = np.unravel_index( np.argmax(votes_heart), img.shape )

    landmarks = irtk.zeros(hough_header,dtype='uint8')
    landmarks[p_left_lung[0],p_left_lung[1],p_left_lung[2]] = 1
    landmarks[p_right_lung[0],p_right_lung[1],p_right_lung[2]] = 2
    landmarks[p_heart[0],p_heart[1],p_heart[2]] = 3
    if ( 0 <= p_liver[0] < landmarks.shape[0] and
         0 <= p_liver[1] < landmarks.shape[1] and
         0 <= p_liver[2] < landmarks.shape[2] ):
        landmarks[p_liver[0],p_liver[1],p_liver[2]] = 4
    if ( 0 <= p_brain[0] < landmarks.shape[0] and
         0 <= p_brain[1] < landmarks.shape[1] and
         0 <= p_brain[2] < landmarks.shape[2] ):
        landmarks[p_brain[0],p_brain[1],p_brain[2]] = 5

    landmarks = irtk.landmarks_to_spheres(landmarks, r=10)
    irtk.imwrite(args.output + "/landmarks.nii.gz",landmarks)
    
    irtk.imwrite(args.output + "/votes_blurred_left_lung.nii.gz",
                 irtk.Image(votes_left_lung,hough_header))
    irtk.imwrite(args.output + "/votes_blurred_right_lung.nii.gz",
                 irtk.Image(votes_right_lung,hough_header))
    irtk.imwrite(args.output + "/votes_blurred_heart.nii.gz",
                 irtk.Image(votes_heart,hough_header))
    irtk.imwrite(args.output + "/votes_blurred_liver.nii.gz",
                 irtk.Image(votes_liver,hough_header))

    if args.back_proj:
        back_proj_left_lung =  hough_votes_backprojection(np.ascontiguousarray(offsets_left_lung.astype('int32')),
                                                          votes_left_lung,
                                                          padding=args.padding)
        back_proj_right_lung =  hough_votes_backprojection(np.ascontiguousarray(offsets_right_lung.astype('int32')),
                                                           votes_right_lung,
                                                           padding=args.padding )
        back_proj_heart = hough_votes_backprojection(np.ascontiguousarray(offsets_heart.astype('int32')),
                                                     votes_heart,
                                                     padding=args.padding )
        back_proj_liver = hough_votes_backprojection(np.ascontiguousarray(offsets_liver.astype('int32')),
                                                     votes_liver,
                                                     padding=args.padding )
        irtk.imwrite(args.output + "/back_proj_left_lung.nii.gz",
                     irtk.Image(back_proj_left_lung,hough_header))
        irtk.imwrite(args.output + "/back_proj_right_lung.nii.gz",
                     irtk.Image(back_proj_right_lung,hough_header))
        irtk.imwrite(args.output + "/back_proj_heart.nii.gz",
                     irtk.Image(back_proj_heart,hough_header))
        irtk.imwrite(args.output + "/back_proj_liver.nii.gz",
                     irtk.Image(back_proj_liver,hough_header))

    votes_left_lung[landmarks!=1] = 0
    votes_right_lung[landmarks!=2] = 0
    votes_heart[landmarks!=3] = 0
    votes_liver[landmarks!=4] = 0
    
    back_proj_left_lung =  hough_votes_backprojection(np.ascontiguousarray(offsets_left_lung.astype('int32')),
                                                      votes_left_lung,
                                                      padding=args.padding)
    back_proj_right_lung =  hough_votes_backprojection(np.ascontiguousarray(offsets_right_lung.astype('int32')),
                                                       votes_right_lung,
                                                      padding=args.padding )
    back_proj_heart = hough_votes_backprojection(np.ascontiguousarray(offsets_heart.astype('int32')),
                                                 votes_heart,
                                                 padding=args.padding )
    back_proj_liver = hough_votes_backprojection(np.ascontiguousarray(offsets_liver.astype('int32')),
                                                 votes_liver,
                                                 padding=args.padding ) 

    final_seg = irtk.zeros( img.get_header(), dtype='uint8' )
    final_seg[back_proj_left_lung>0] = 1
    final_seg[back_proj_right_lung>0] = 2
    final_seg[back_proj_heart>0] = 3
    final_seg[back_proj_liver>0] = 4
    irtk.imwrite(args.output + "/final_seg.nii.gz",
                  irtk.Image(final_seg,img.get_header()))
    
print pr_left_lung, pr_right_lung, pr_liver, d
