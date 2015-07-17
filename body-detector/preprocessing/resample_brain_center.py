#!/usr/bin/python

import irtk
import numpy as np
import scipy.ndimage as nd
from glob import glob
import csv
import os

from joblib import Parallel, delayed

import sys
sys.path.append("../lib")
from libdetector import *

data_folder = "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/full_body_segmentations"

output_folder = "/vol/bitbucket/kpk09/detector/"

all_files = glob(data_folder + "/*_seg.nii.gz")

all_ga = {}
reader = csv.reader( open( data_folder + "/../melissa_ga.tsv", "rb"), delimiter="\t" )
for patient_id, ga in reader:
    all_ga[patient_id] = float(ga)

def run(f):
    patient_id = os.path.basename(f)[:-len("_seg.nii.gz")]

    print "PATIENT_ID",patient_id

    f_img = data_folder + "/" + patient_id + ".nii"
    if not os.path.exists(f_img):
        f_img += ".gz"
    seg = irtk.imread( f, dtype='float32', force_neurological=True )
    img = irtk.imread( f_img, dtype='float32', force_neurological=True )

    img = irtk.Image(nd.median_filter(img.view(np.ndarray),(3,5,5)),img.get_header())
    
    ga = all_ga[patient_id]

    scale = get_CRL(ga)/get_CRL(30.0)

    OFD = get_OFD(30.0)
    BPD = get_BPD(30.0)
    CRL = get_CRL(30.0)

    brain_center = seg.ImageToWorld( np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                                                 dtype='float32')[::-1] )

    header = img.get_header()
    header['origin'][:3] = brain_center
    header['pixelSize'][:3] = 1.0*scale
    header['dim'][0] = CRL
    header['dim'][1] = CRL
    header['dim'][2] = CRL
    
    img = img.transform( target=header, interpolation="bspline" )
    seg = seg.transform( target=header, interpolation="nearest" )

    img[img<1.0] = 0
    
    irtk.imwrite(output_folder + "brain_center/"+patient_id+"_img.nii.gz",img)
    irtk.imwrite(output_folder + "brain_center/"+patient_id+"_seg.nii.gz",seg)
    
    return
    
Parallel(n_jobs=10)(delayed(run)(f) for f in all_files )
