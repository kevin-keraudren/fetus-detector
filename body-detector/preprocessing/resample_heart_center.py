#!/usr/bin/python
"""
Resample and align fetuses centered on the heart.
"""

import irtk
import numpy as np
import scipy.ndimage as nd
from glob import glob
import csv
import os

from joblib import Parallel, delayed
from skimage import morphology

import sys
sys.path.append("../../commonlib")
from fetal_anatomy import *

def align_heart(img,labels):

    BPD = get_BPD(30.0)
    CRL = get_CRL(30.0)

    brain_center = labels.ImageToWorld( np.array(nd.center_of_mass( (labels == 2).view(np.ndarray) ),
                                                  dtype='float32')[::-1] )

    heart_center =  labels.ImageToWorld( np.array(nd.center_of_mass( (labels == 5).view(np.ndarray) ),
                                                  dtype='float32')[::-1] )

    lungs_center =  labels.ImageToWorld( np.array(nd.center_of_mass(np.logical_or(labels == 3,
                                                                                   labels == 4 ).view(np.ndarray)
                                                                    ),
                                                  dtype='float32')[::-1] )

    left_lung = labels.ImageToWorld( np.array(nd.center_of_mass( (labels == 3).view(np.ndarray) ),
                                              dtype='float')[::-1] )
    right_lung = labels.ImageToWorld( np.array(nd.center_of_mass( (labels == 4).view(np.ndarray) ),
                                               dtype='float')[::-1] )
    
    u = brain_center - heart_center
    #v = lungs_center - heart_center
    v = right_lung - left_lung
    
    u /= np.linalg.norm(u)

    v -= np.dot(v,u)*u
    v /= np.linalg.norm(v)

    w = np.cross(u,v)
    w /= np.linalg.norm(w)
    
    # v = np.cross(w,u)
    # v /= np.linalg.norm(v)

    header = img.get_header()
    header['orientation'][0] = u
    header['orientation'][1] = v
    header['orientation'][2] = w

    header['origin'][:3] = heart_center
    
    header['dim'][0] = CRL
    header['dim'][1] = CRL
    header['dim'][2] = CRL
    
    new_img = img.transform( target=header, interpolation="bspline" )
    new_labels = labels.transform( target=header, interpolation="nearest" )

    return new_img, new_labels

data_folder = "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/full_body_segmentations"
output_folder = "/vol/bitbucket/kpk09/detector"

all_landmarks = glob(data_folder+"/*_seg.nii.gz")

all_ga = {}
reader = csv.reader( open( data_folder + "/../melissa_ga.tsv", "rb"), delimiter="\t" )
for patient_id, ga in reader:
    all_ga[patient_id] = float(ga)

all_iugr = {}
reader = csv.reader( open( "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/iugr.tsv", "rb"), delimiter="\t" )
skip = True
for patient_id, iugr, iugr_severity in reader:
    if skip:
        skip = False
        continue
    all_iugr[patient_id] = (int(iugr), int(iugr_severity))
        
def run(f):
    patient_id = os.path.basename(f)[:-len("_seg.nii.gz")]

    # if patient_id != "1585":
    #     return
    
    print "PATIENT_ID",patient_id
    
    f_img = data_folder + "/" + patient_id + ".nii"
    if not os.path.exists(f_img):
        f_img += ".gz"
        
    seg = irtk.imread( f, dtype='float32', force_neurological=True )
    img = irtk.imread( f_img, dtype='float32', force_neurological=True )
    
    img = irtk.Image(nd.median_filter(img.view(np.ndarray),(3,5,5)),img.get_header())

    ga = all_ga[patient_id]

    scale = get_CRL(ga)/get_CRL(30.0)
    # if all_iugr[patient_id][0] == 1:
    #     scale = get_CRL(ga,0.02) / get_CRL(30,0.5)
    # else:
    #     scale = get_CRL(ga,0.5) / get_CRL(30,0.5)

    ## if using the weight, then take the cubic root
    # if all_iugr[patient_id][0] == 1:
    #     scale = (get_weight(ga,0.02) / get_weight(30,0.5)) ** (1.0/3.0)
    # else:
    #     scale = (get_weight(ga,0.5) / get_weight(30,0.5)) ** (1.0/3.0)
    
    # seg = seg.resample( 1.0*scale, interpolation='nearest')
    # img = img.resample( 1.0*scale, interpolation='bspline' )

    seg = seg.resample( 1.0, interpolation='nearest')
    img = img.resample( 1.0, interpolation='bspline' )

    #irtk.imwrite(output_folder + "/heart_center_weight/"+patient_id+"_seg.nii.gz",seg)
    
    img_abdomen, labels_abdomen = align_heart( img, seg )

    irtk.imwrite(output_folder + "/heart_center_no_resizing/"+patient_id+"_img.nii.gz",img_abdomen)
    irtk.imwrite(output_folder + "/heart_center_no_resizing/"+patient_id+"_seg.nii.gz",labels_abdomen)
    return

    
Parallel(n_jobs=20)(delayed(run)(f) for f in all_landmarks )
