#!/usr/bin/python

import numpy as np
import irtk
import scipy.ndimage as nd
from skimage import morphology

from scipy.spatial import ConvexHull
from irtk.vtk2irtk import voxellise

import argparse

parser = argparse.ArgumentParser(
    description='' )
parser.add_argument( '--proba', type=str, required=True )
parser.add_argument( '--seg', type=str, default=None )
parser.add_argument( '--img', type=str, required=True )
parser.add_argument( '--output', type=str, required=True )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--remove_small_objects', type=int, default=0 )
parser.add_argument( '--dilate', type=int, default=0 )
parser.add_argument( '--sigma', type=float, default=10.0 )
parser.add_argument( '-l', type=float, default=20.0 )
parser.add_argument( '--select', action="store_true", default=False )
parser.add_argument( '--hull', action="store_true", default=False )
args = parser.parse_args()
print args

proba = irtk.imread( args.proba, dtype='float32', force_neurological=True )
img = irtk.imread( args.img, dtype='float32', force_neurological=True ).rescale(0,1000)

# for i in range(proba.shape[0]):
#     proba[i] /= proba[i].max()

proba[0] += proba[4] + proba[3]
proba[1] += proba[2]
# #proba[2] = proba[3]

header = proba.get_header()
header['dim'][3] = 2

proba = proba[:2]
proba = irtk.Image(proba,header)#.saturate(10,90).rescale(0,1)

proba[1] = nd.gaussian_filter(proba[1],1.0)
proba[1] /= proba[1].max()
proba[0] = 1.0 - proba[1]

# #proba[proba>1] = 1

irtk.imwrite( "proba_debug.nii.gz", proba )

# seg = irtk.zeros( img.get_header(), dtype='int32' )
# for i in [1,2]:
#     seg[proba[i] > 0.5] = i

if args.seg is None:
    seg_init = irtk.Image( np.argmax(proba,axis=0).astype('int32'),
                      img.get_header() )
else:
    seg_init = irtk.imread( args.seg, dtype='int32', force_neurological=True )
    seg_init[seg_init>2] = 0
    seg_init[seg_init==2] = 1

irtk.imwrite("seg_debug.nii.gz",seg_init)

seg = irtk.crf( img,
                seg_init,
                proba,
                l=args.l,
                sigma=args.sigma ) # 30,10 # 10,5

print 1, seg.shape, seg_init.shape

print np.unique(seg)

if args.remove_small_objects > 0:
    seg = irtk.Image( morphology.remove_small_objects(seg.view(np.ndarray).astype('bool'),
                                                      min_size=args.remove_small_objects).astype('uint8'),
                      seg.get_header() )

print 2, seg.shape, seg_init.shape
    
if args.select:
    seg = irtk.Image( morphology.watershed(seg,seg_init,mask=seg ),
                      seg.get_header() )

print 3, seg.shape, seg_init.shape

if args.dilate > 0:
    ball = morphology.ball(args.dilate)
    seg = irtk.Image( nd.binary_dilation( seg.view(np.ndarray),
                                          structure=ball,
                                          #iterations=args.dilate,
                                          ).astype('uint8'),
                      seg.get_header() )

print 4, seg.shape, seg_init.shape

if args.hull:
    #seg = morphology.convex_hull_image(seg)
    #ZYX = np.where(seg)
    ZYX = np.transpose(np.nonzero(seg))
    print ZYX
    pts = seg.ImageToWorld( ZYX[:,::-1] )
    hull = ConvexHull(pts,qhull_options="Qx Qs QbB QJ")
    print hull.points, hull.simplices
    seg = voxellise(  hull.points, hull.simplices,
            header=seg.get_header() )

print 5, seg.shape, seg_init.shape
            
irtk.imwrite( args.output, seg )
