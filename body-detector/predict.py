#!/usr/bin/python

import irtk
import numpy as np
import os
import shutil

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

import joblib
from joblib import Parallel, delayed
import cPickle as pickle

import argparse

from skimage.feature import peak_local_max
import subprocess

from libdetector import *
from helpers import *

import gc
import scipy.sparse as sp

def get_orientation( center, coords, brain_jitter=10 ):
    u = ( ( center +
            np.random.randint(-brain_jitter, brain_jitter+1,
                              size=(coords.shape[0],3) ))
          - coords )
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
parser.add_argument( "--detector2", type=str, required=True, action=PathExists )
parser.add_argument( "--shape_model", type=str, required=True, action=PathExists )
parser.add_argument( "--output", type=str, required=True )
parser.add_argument( "--output2", type=str, required=True )
parser.add_argument( '--brain_center', type=float, nargs=3, required=True )
# parameters with default values
parser.add_argument( "--shape_optimisation", action="store_true", default=False )
parser.add_argument( '--n_jobs', type=int, default=20 )
parser.add_argument( '--chunk_size', type=int, default=int(3e6) )
parser.add_argument( '--fast', type=float, default=0.0 )
parser.add_argument( '--narrow_band', action="store_true", default=False )
parser.add_argument( '--brain_jitter', type=int, default=10 )
parser.add_argument( '--theta', type=int, default=90 )
parser.add_argument( '--padding', type=int, default=10 )
parser.add_argument( '--distribute', type=int, default=-1 )
parser.add_argument( '-l', type=float, default=0.5 )
parser.add_argument( '--rw', action="store_true", default=False )
parser.add_argument( '--selective', action="store_true", default=False )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--not_centered', action="store_true", default=False )
parser.add_argument( '--random', action="store_true", default=False )
args = parser.parse_args()

print args

if not os.path.exists(args.output):
    os.makedirs(args.output)

nb_labels = 2

print "loading detectors..."

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

clf_heart = joblib.load( args.detector + '/clf_heart' )
reg_heart = joblib.load( args.detector + '/reg_heart' )

clf_heart.set_params(n_jobs=args.n_jobs)
reg_heart.set_params(n_jobs=args.n_jobs)

print "done loading detectors"
print "preprocessing..."

img = irtk.imread( args.input, dtype='float32', force_neurological=True )

grad = irtk.Image(nd.gaussian_gradient_magnitude( img, 0.5 ),
              img.get_header())

sat = integral_image(img)
sat_grad = integral_image(grad)

blurred_img = nd.gaussian_filter(img,0.5)
gradZ = nd.sobel( blurred_img, axis=0 ).astype('float32')
gradY = nd.sobel( blurred_img, axis=1 ).astype('float32')
gradX = nd.sobel( blurred_img, axis=2 ).astype('float32')

irtk.imwrite(args.output + "/img.nii.gz", img)
irtk.imwrite(args.output + "/grad.nii.gz", grad)

print "done preprocessing"

del blurred_img
del grad

args.brain_center = np.array(args.brain_center, dtype='float32')

print args.brain_center

narrow_band = get_narrow_band( img.shape,
                               args.brain_center,
                               r_min=40,
                               r_max=120 )
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
    print "chunk size too large, setting it to", args.chunk_size

header['dim'][3] = nb_labels
proba = irtk.zeros(header,dtype='float32')

u,v0,w0 = get_orientation(args.brain_center,coords,args.brain_jitter)

# If the location of the brain is unknown,
# we can try random orientations
if args.random:
    u,v0,w0 =  get_random_orientation(coords.shape[0])

best_proba = irtk.zeros(header,dtype='float32')
min_proba = irtk.zeros(header,dtype='float32')
if args.fast:
    best_proba[ 0,
                coords2[:,0],
                coords2[:,1],
                coords2[:,2]] = np.finfo('float32').max
    min_proba[ 1,
                coords2[:,0],
                coords2[:,1],
                coords2[:,2]] = np.finfo('float32').max    
else:
    best_proba[ 0,
                coords[:,0],
                coords[:,1],
                coords[:,2]] = np.finfo('float32').max
    min_proba[ 1,
                coords[:,0],
                coords[:,1],
                coords[:,2]] = np.finfo('float32').max
    
best_v = v0.copy()
best_w = w0.copy()

print "doing classification"

for theta in xrange(0,360,args.theta):
    print "tetha =", theta
    v = np.cos(float(theta)/180*np.pi)*v0 + np.sin(float(theta)/180*np.pi)*w0
    w = np.cos(float(theta+90)/180*np.pi)*v0 + np.sin(float(theta+90)/180*np.pi)*w0
    for i in xrange(0,coords.shape[0],args.chunk_size):
        j = min(i+args.chunk_size,coords.shape[0])
        # pre-allocating memory
        x = np.zeros( (j-i,offsets1.shape[0]+offsets3.shape[0]+offsets5.shape[0]+1), dtype='float32' )
        if args.not_centered:
             x1 = get_block_comparisons_cpp( sat, coords[i:j],
                                             offsets1, sizes1,
                                             offsets2, sizes2,n_jobs=args.n_jobs )
             x2 = get_block_comparisons_cpp( sat_grad, coords[i:j],
                                             offsets3, sizes3,
                                             offsets4, sizes4, n_jobs=args.n_jobs)
             x_grad1 = get_grad( coords[i:j].astype('int32'), offsets5.astype('int32'),
                                 gradZ.astype('float32'), gradY.astype('float32'), gradX.astype('float32') )
             x_grad2 = get_grad( coords[i:j].astype('int32'), offsets6.astype('int32'),
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
            
            x[:,feature_offset:feature_offset+offsets3.shape[0]] = get_block_comparisons_cpp_uvw(
                sat_grad,
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

        x[:,feature_offset] = np.linalg.norm( coords[i:j] - args.brain_center, axis=1 )
        
        pr = clf_heart.predict_proba(x)

        if args.fast:
            selection = pr[:,0] < best_proba[0,
                                             coords2[i:j,0],
                                             coords2[i:j,1],
                                             coords2[i:j,2]]
            best_v[i:j][selection] = v[i:j][selection]
            best_w[i:j][selection] = w[i:j][selection]
            for dim in xrange(2):
                best_proba[ dim,
                            coords2[i:j,0][selection],
                            coords2[i:j,1][selection],
                            coords2[i:j,2][selection] ] = pr[:,dim][selection]

            selection = pr[:,1] < min_proba[1,
                                            coords2[i:j,0],
                                            coords2[i:j,1],
                                            coords2[i:j,2]]
            for dim in xrange(2):
                min_proba[ dim,
                            coords2[i:j,0][selection],
                            coords2[i:j,1][selection],
                            coords2[i:j,2][selection] ] = pr[:,dim][selection]   
        else:
            selection = pr[:,0] < best_proba[0,
                                             coords[i:j,0],
                                             coords[i:j,1],
                                             coords[i:j,2]]
            best_v[i:j][selection] = v[i:j][selection]
            best_w[i:j][selection] = w[i:j][selection]
            for dim in xrange(2):
                best_proba[ dim,
                            coords[i:j,0][selection],
                            coords[i:j,1][selection],
                            coords[i:j,2][selection] ] = pr[:,dim][selection]

            selection = pr[:,1] < min_proba[1,
                                            coords[i:j,0],
                                            coords[i:j,1],
                                            coords[i:j,2]]
            for dim in xrange(2):
                min_proba[ dim,
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
    min_proba = min_proba.transform(target=header)

irtk.imwrite( args.output + "/proba.nii.gz", proba )

irtk.imwrite( args.output + "/min_proba.nii.gz", min_proba )

seg = np.argmax( proba, axis=0)
seg = irtk.Image( seg,
                  img.get_header() ).astype('uint8')

irtk.imwrite( args.output + "/seg.nii.gz", seg )

# Regression
print "doing regression"
header = img.get_header()
header['dim'][3] = 3
offsets_heart = irtk.ones(header,dtype='float32') * np.finfo('float32').max
for l, forest, offsets in zip( [1],
                               [reg_heart],
                               [offsets_heart] ):  

    if args.theta < 360 and args.selective:
        selection = np.logical_and( proba[l,
                                          coords[:,0],
                                          coords[:,1],
                                          coords[:,2]]>0.5,
                                    min_proba[l,
                                              coords[:,0],
                                              coords[:,1],
                                              coords[:,2]]<0.5 )
    else:
        selection = proba[l,
                          coords[:,0],
                          coords[:,1],
                          coords[:,2]] > 0.5
        
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
            
irtk.imwrite( args.output + "/offsets_heart.nii.gz",
              offsets_heart )

hough_header = img.get_header()
hough_header['dim'][:3] += 2*args.padding

votes_heart = hough_votes(np.ascontiguousarray(np.round(offsets_heart).astype('int32')),
                            proba[1], padding=args.padding )

irtk.imwrite(args.output + "/votes_heart.nii.gz",
             irtk.Image(votes_heart,hough_header))

# mask using narrow_band
if args.padding:
    # update the narrow band without forgetting to
    # translate the brain_center
    narrow_band = get_narrow_band( votes_heart.shape,
                                   args.brain_center+args.padding,
                                   r_min=40,
                                   r_max=120 )

votes_heart[narrow_band==0] = 0

votes_heart, scale_heart = rescale( nd.gaussian_filter(votes_heart, 2.0), dtype='float32')

irtk.imwrite(args.output + "/votes_blurred_heart.nii.gz",
             irtk.Image(votes_heart,hough_header))
        
def score_candidate( h, b, output=None, verbose=False, debug=True):
    cmd = [ 'python', os.path.dirname(os.path.abspath(__file__))+'/predict2.py',
            '--input', args.input,
            '--detector', args.detector2,
            '--n_jobs', args.n_jobs,
            '--chunk_size', args.chunk_size,
            '--shape_model', args.shape_model,
            '--theta', str(args.theta),
            '--narrow_band',
            '--selective',
            '--padding', str(args.padding),
            '-l', str(args.l),
            '--brain_center'] + map(str,b) + \
           ['--heart_center'] + map(str,h)
    if output is None:
        cmd.append( '--score')
    else:
        cmd.extend( ['--output',output])

    if args.fast > 0:
        cmd.extend( ['--fast',args.fast])
        
    if args.random:
        cmd.append( '--random')

    if verbose:
        cmd.append('--verbose')
        
    cmd = map(str,cmd)
    if debug:
        print " ".join(cmd)
        
    p = subprocess.Popen( cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE )
    output, err = p.communicate()
    rc = p.returncode

    if debug:
        print output
        print err

    if not verbose:
        # pr_left_lung, pr_right_lung, pr_liver, d
        return map( float, output.rstrip().split(' ') )
    else:
        print output
        return
        
if args.shape_optimisation:
    shape_model = pickle.load( open( args.shape_model, 'rb' ) )
    
    local_max = peak_local_max( votes_heart,
                                min_distance=10,
                                threshold_rel=0.2,
                                exclude_border=False,
                                indices=True,
                                num_peaks=5 )

    tmp_output = args.output + "/tmp_predictions"
    if not os.path.exists(tmp_output):
        os.makedirs(tmp_output)

    # tmp_scores = []
    # for p in local_max:
    #     tmp_scores.append( score_candidate(p-args.padding,args.brain_center) )
    tmp_scores = Parallel(n_jobs=args.distribute,backend="threading")(delayed(score_candidate)(p-args.padding,
                                                                       args.brain_center,
                                                                       output=tmp_output+"/"+str(i))
                                              for i,p in enumerate(local_max) )
    tmp_scores = np.array( tmp_scores )

    for i in xrange(3):
        m = tmp_scores[:,i].max()
        if m != 0:
            tmp_scores[:,i] /= m

    scores = []
    for p, tmp_s in zip( local_max, tmp_scores):
        s = args.l*votes_heart[p[0],p[1],p[2]] + (1.0-args.l)*shape_proba_brain(p,args.brain_center+args.padding, shape_model)
        s += args.l*tmp_s[:3].sum() + (1.0-args.l)*tmp_s[3]
        print p, s
        scores.append(s)

    scores = np.array(scores)
    score = scores.max()
    sys.stderr.write( str(score) + "\n" )
    best_candidate = np.argmax(scores)
    p = local_max[best_candidate]

    if os.path.exists(args.output2):
        shutil.rmtree(args.output2)
    os.renames(tmp_output+"/"+str(best_candidate), args.output2)
    if os.path.exists(tmp_output):
        shutil.rmtree(tmp_output)
    
else:
    p = np.unravel_index(np.argmax(votes_heart), votes_heart.shape )
    score_candidate( p-args.padding,
                     args.brain_center,
                     output=args.output2,
                     verbose=True )
    
landmark = irtk.zeros(hough_header,dtype='uint8')
landmark[p[0],p[1],p[2]] = 1

landmark = irtk.landmarks_to_spheres(landmark, r=10)
irtk.imwrite(args.output + "/detected_heart.nii.gz",landmark)





# for each landmark, run predict2, keep best result

if args.rw:
    img_filename = args.output2 + "/img.nii.gz"
    seg_filename = args.output2 + "/final_seg.nii.gz"
    rw_filename = args.output2 + "/rw.nii.gz"
    cmd = [ "python", os.path.dirname(__file__)+"/segment_random_walker.py",
            "--img", img_filename,
            "--seg", seg_filename,
            "--output", rw_filename ]

    print ' '.join(cmd)

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    
    print out
    print err
