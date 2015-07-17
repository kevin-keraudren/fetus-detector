#!/usr/bin/python

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

img_folder = "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/full_body_segmentations"
seg_folder = "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/full_body_segmentations_lcc"
output_folder = "/vol/bitbucket/kpk09/detector"

all_landmarks = glob(seg_folder+"/*_seg.nii.gz")

all_ga = {}
reader = csv.reader( open( img_folder + "/../melissa_ga.tsv", "rb"), delimiter="\t" )
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
    
    f_img = img_folder + "/" + patient_id + ".nii"
    if not os.path.exists(f_img):
        f_img += ".gz"
        
    seg = irtk.imread( f, dtype='float32', force_neurological=True )
    img = irtk.imread( f_img, dtype='float32', force_neurological=True )
    
    img = irtk.Image(nd.median_filter(img.view(np.ndarray),(3,5,5)),img.get_header())

    ga = all_ga[patient_id]

    scale = get_CRL(ga)/get_CRL(30.0)

    # if all_iugr[patient_id][0] == 1:
    #     scale = (get_weight(ga,0.02) / get_weight(30,0.5)) ** (1.0/3.0)
    # else:
    #     scale = (get_weight(ga,0.5) / get_weight(30,0.5)) ** (1.0/3.0)
    
    seg = seg.resample( 1.0*scale, interpolation='nearest')
    img = img.resample( 1.0*scale, interpolation='bspline' )
    
    irtk.imwrite(output_folder + "/data_resampled_weight/"+patient_id+"_img.nii.gz",img)
    irtk.imwrite(output_folder + "/data_resampled_weight/"+patient_id+"_seg.nii.gz",seg)

    return

    
Parallel(n_jobs=10)(delayed(run)(f) for f in all_landmarks )
