#!/usr/bin/python

import csv
import pickle
from glob import glob
import os
import irtk
import numpy as np
import scipy.ndimage as nd
from skimage import restoration
from skimage import morphology
from joblib import Parallel, delayed

import sys
sys.path.insert(1,"../commonlib")
from fetal_anatomy import *

ga_file = "/vol/vipdata/data/fetal_data/motion_correction/ga.tsv"
raw_folder = "/vol/vipdata/data/fetal_data/motion_correction/original_scans/"
detection_folder = "/vol/vipdata/data/fetal_data/OUTPUT/whole_body_shape_padding50"
reconstruction_folder = "/vol/vipdata/data/fetal_data/OUTPUT/thorax_reconstructionGPU"

reconstruction_binary = "/vol/medic02/users/kpk09/github/irtk/build/bin/reconstructionMasking"
reconstruction_binary = "/vol/medic02/users/kpk09/github/fetalReconstruction/source/bin/reconstruction_GPU2"
f_template = "/vol/medic02/users/kpk09/gitlab/fetus-detector/body-detector/img/seg_template.nii.gz"

reader = csv.reader( open( ga_file, "rb"), delimiter=" " )
all_ga = {}
for patient_id, ga in reader:
    all_ga[patient_id] = ga

script = open( "reconstruct.sh", "wb")
script.write("#!/bin/bash\n\n")
script.write("set -x\n")
script.write("set -e\n\n")

rreg = "/vol/medic02/users/kpk09/github/irtk/build/bin/rreg"


def mask_data(f):
    file_id = f.split('/')[-3]
    seg = irtk.imread(f,force_neurological=True) > 0

    r = 10
    x_min,y_min,z_min,x_max,y_max,z_max = seg.bbox()
    seg = seg[max(0,z_min-3*r):min(z_max+3*r+1,seg.shape[0]),
              max(0,y_min-3*r):min(y_max+3*r+1,seg.shape[1]),
              max(0,x_min-3*r):min(x_max+3*r+1,seg.shape[2])]
    ball = morphology.ball( 5 )
    seg = irtk.Image( nd.binary_dilation(seg,ball), seg.get_header() )
    ball = morphology.ball( r )
    seg = irtk.Image( nd.binary_closing(seg,ball), seg.get_header() )
    
    seg = seg.bbox(crop=True)
        
    seg_file = output_dir + '/seg_' + file_id + ".nii.gz"
    irtk.imwrite( seg_file, seg )

def align_to_template(f,f_template,output_folder,ga):
    file_id = f.split('/')[-3]
    landmarks = irtk.imread(f,force_neurological=True)
    scale = get_CRL(ga)/get_CRL(30.0)
    template = irtk.imread(f_template,force_neurological=True)
    template.header['pixelSize'][:3] /= scale
    points = []
    points_template = []
    for i,j in zip( [2,8,3,4,5],
                    [5,4,1,2,3] ):
       points_template.append( template.ImageToWorld( nd.center_of_mass(template.view(np.ndarray)==i)[::-1] ) )
       points.append( landmarks.ImageToWorld( nd.center_of_mass(landmarks.view(np.ndarray)==j)[::-1] ) )

    t,rms = irtk.registration_rigid_points( np.array(points),
                                            np.array(points_template),
                                            rms=True )
    print "RMS: ", rms
    t.invert().write( output_folder + '/' + file_id + '.dof' )
    landmarks = landmarks.transform(t,target=template)
    irtk.imwrite( output_folder + '/landmarks_' + file_id + '.nii.gz',landmarks )
    return t

def crop_data(f,mask,t):
    file_id = os.path.basename(f).split('.')[0]
    
    img = irtk.imread( f, dtype='float32',force_neurological=True )
    seg = mask.transform(t.invert(), target=img,interpolation='nearest')
    x_min,y_min,z_min,x_max,y_max,z_max = seg.bbox()
    seg = seg[z_min:z_max+1,
              y_min:y_max+1,
              x_min:x_max+1]
    img = img[z_min:z_max+1,
              y_min:y_max+1,
              x_min:x_max+1].rescale(0,1000) + 1.0 # +1 to avoid zeros in the heart
    img_file = output_dir + '/img_' + file_id + ".nii.gz"
    irtk.imwrite( img_file, img )
    for z in range(img.shape[0]):
        scale = img[z].max()
        img[z] = restoration.nl_means_denoising(img[z].rescale(0.0,1.0).view(np.ndarray),
                                                fast_mode=False,
                                                patch_size=5,
                                                patch_distance=7,
                                                h=0.05,
                                                multichannel=False)
        img[z] *= scale
    img[seg==0] = 0
    masked_file = output_dir + '/masked_' + file_id + ".nii.gz"
    irtk.imwrite( masked_file, img )

def get_corners(img):
    pts = np.array([[0,0,0],
                    [0,0,img.header['dim'][2]],
                    [0,img.header['dim'][1],0],
                    [img.header['dim'][0],0,0],
                    [0,img.header['dim'][1],img.header['dim'][2]],
                    [img.header['dim'][0],0,img.header['dim'][2]],
                    [img.header['dim'][0],img.header['dim'][1],0],
                    [img.header['dim'][0],img.header['dim'][1],img.header['dim'][2]]])
    pts = img.ImageToWorld( pts )
    return pts
        
def create_mask_from_all_masks(f_lists,transformations,ga,resolution=0.75):
    points = []
    for f, t in zip(f_lists,transformations):
        m = irtk.imread(f,force_neurological=True)
        points.extend( t.apply(get_corners(m)) )

    points = np.array(points,dtype='float64')
    
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    pixelSize = [resolution, resolution, resolution, 1]
    orientation = np.eye( 3, dtype='float64' )
    origin = [ x_min + (x_max - x_min)/2,
               y_min + (y_max - y_min)/2,
               z_min + (z_max - z_min)/2,
               0 ]
    dim = [ (x_max - x_min)/resolution,
            (y_max - y_min)/resolution,
            (z_max - z_min)/resolution,
            1 ]

    header = irtk.new_header( pixelSize=pixelSize,
                orientation=orientation,
                origin=origin,
                dim=dim )

    mask = irtk.zeros( header, dtype='float32' )

    for f, t in zip( f_lists, transformations ):
        m = irtk.imread(f,force_neurological=True).transform(t, target=mask,interpolation="linear")
        mask += m

    irtk.imwrite( "debug_mask1.nii.gz", mask)
    
    mask = irtk.Image( nd.gaussian_filter( mask, 0.5 ),
                       mask.get_header() )

    irtk.imwrite( "debug_mask2.nii.gz", mask)

    mask = (mask > 0).bbox(crop=True).astype('uint8')

    scale = get_CRL(ga)/get_CRL(30.0)
    template = irtk.imread(f_template,force_neurological=True)
    template.header['pixelSize'][:3] /= scale
    
    template = template.transform(target=mask,interpolation='nearest')
    mask[template==0] = 0

    irtk.imwrite( "debug_template.nii.gz", template)
    irtk.imwrite( "debug_mask3.nii.gz", mask)

    return mask


for patient_id in all_ga:
    glob_pattern = detection_folder + "/" + patient_id + "_*/prediction_2/rw10.nii.gz"
    print glob_pattern
    files = glob( glob_pattern )
    if len(files) == 0:
        print "not doing patient",patient_id
        continue

    if patient_id != '3115':
        continue

    output_dir = reconstruction_folder + '/' + patient_id
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ga = float(all_ga[patient_id])

    seg_files = []
    img_files = []
    masked_files = []
    landmark_files = []
    original_files = []
    transformation_files = []
    for f in files:
        file_id = f.split('/')[-3]
        original_file = raw_folder+'/'+file_id+".nii"
        original_files.append(original_file)
        seg_file = output_dir + '/seg_' + file_id + ".nii.gz"
        seg_files.append(seg_file)
        img_file = output_dir + '/img_' + file_id + ".nii.gz"
        img_files.append(img_file)
        masked_file = output_dir + '/masked_' + file_id + ".nii.gz"
        masked_files.append(masked_file)
        landmark_file = detection_folder + "/" + file_id + "/prediction_2/landmarks.nii.gz"
        landmark_files.append( landmark_file )
        transformation_file = output_dir + '/' + file_id + ".dof"
        transformation_files.append(transformation_file)
        
    Parallel(n_jobs=-1)(delayed(mask_data)(f) for f in files)
    transformations = Parallel(n_jobs=-1)(delayed(align_to_template)(f,f_template,output_dir,ga)
                                          for f in landmark_files)

    mask_file = output_dir + '/' + patient_id + "_mask.nii.gz"
    mask = create_mask_from_all_masks(seg_files, transformations, ga,resolution=0.75)
    irtk.imwrite( mask_file, mask )

    Parallel(n_jobs=-1)(delayed(crop_data)(f,mask,t) for f,t in zip(original_files,transformations))
    
    cmd = ( ["/usr/bin/time",
             "--verbose",
             "--output",
             patient_id + "_masking.time",
             reconstruction_binary]
            + [ "-o", "reconstruction_" + patient_id + ".nii.gz",
                "-m", mask_file,
                "-i"]
            + img_files
            # + ["-dofin"] + transform
            + ['-t'] + transformation_files
            + ["--log_prefix", patient_id + "_masking",
               "--smooth_mask", str(4),
              # "-resolution", str(0.8),
               "--debug_gpu",
               "--debug", "1",
               "-d", "0",
            #   "-not_brain",
               #"-adaptive",
               #"-info", "slice_info_masking.tsv",
               "--lambda", str(0.03),
               "--lastIter", str(0.02),
               "--delta", str(300),
               "--iterations", str(6)]
            )

    script.write( "# " + patient_id + "\n" )
    script.write( "sbatch --mem=30G -c 8 -p gpu --wrap=\"cd " + output_dir + " && " )
    script.write(' '.join(cmd) + "\"\n" )
    script.write( "\n" )
