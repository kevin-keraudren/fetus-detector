#!/usr/bin/python

import sys
import csv
import os

import irtk
import scipy.ndimage as nd
import numpy as np

import subprocess
import argparse

sys.path.append("../commonlib")
from fetal_anatomy import *
from commonlibdetector import *

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( '--filename', type=str, required=True ) 
parser.add_argument( '--ga', type=float, required=True )     
parser.add_argument( '--output_folder', type=str, default="detection_results" )        
args = parser.parse_args()    

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)    
   
def run_detection( filename, ga, output_folder ):
    file_id = os.path.basename(filename).split('.')[0]
    if '_' in os.path.basename(filename):
        patient_id = file_id.split('_')[0]
    else:
        patient_id = file_id
    print patient_id 
    
    # brain detection
    vocabulary = "../brain-detector/trained_model/vocabulary_0.npy"
    mser_detector = "../brain-detector/trained_model/mser_detector_0_LinearSVC"
    mask_file = output_folder +"/" + file_id + "/brain_mask.nii.gz"
    cmd = [ "python",
            "../brain-detector/fetalMask_detection.py",
            filename,
            str(ga),
            mask_file,
            "--classifier", mser_detector,
            "--vocabulary", vocabulary
            ]
    print ' '.join(cmd)
    
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE )
    (out, err) = proc.communicate()

    print out
    print err

    # heart and body detection

    ## preprocessing (resampling+denoising)
    img = irtk.imread( filename, dtype='float32', force_neurological=True )
    img = irtk.Image(nd.median_filter(img.view(np.ndarray),(3,5,5)),img.get_header())

    scale = get_CRL(ga)/get_CRL(30.0)

    img = img.resample( 1.0*scale, interpolation='bspline' )
    
    brain_center = img.WorldToImage( get_center_brain_detection(mask_file) )[::-1]
    
    new_filename = output_folder + "/" + file_id + "/" + os.path.basename(filename)
    
    irtk.imwrite(new_filename,img)
    
    n_jobs = 5

    output_folder1 = output_folder + "/" + file_id + "/prediction_1/"
    output_folder2 = output_folder + "/" + file_id + "/prediction_2"
    detector1 = "trained_model/stage1"
    detector2 = "trained_model/stage2"
    shape_model = "trained_model/stage1/shape_model.pk"

    cmd = [ "python", "predict.py",
            "--input", new_filename,
            "--output", output_folder1,
            "--output2", output_folder2,
            "--detector", detector1,
            "--detector2", detector2,
            "--padding", str(10),
            "--chunk_size", str(int(1e5)),
            "--n_jobs", str(n_jobs),
            "--brain_center"] + map(str,brain_center) + \
            ["--shape_model",shape_model,
             "-l", str(0.5),
             "--shape_optimisation",
             "--narrow_band",
             "--selective"]

    print ' '.join(cmd)

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()

    print out
    print err
   
    return

run_detection( args.filename, args.ga, args.output_folder )
