#!/usr/bin/python

import sys
import csv
import os

import irtk
import scipy.ndimage as nd
import numpy as np

import subprocess

# sys.path.append("lib")
# from libdetector import *

sys.path.append("../commonlib")
from fetal_anatomy import *
from commonlibdetector import *

def get_fold(patient_id):
    for i in range(10):
        testing_file = "/vol/biomedic/users/kpk09/gitlab/fetal-brain-detection/10_folds/testing_"+str(i)+".tsv"
        selected_patients = map( lambda x: x.rstrip(),
                                 open( testing_file, 'rb' ).readlines() )
        if patient_id in selected_patients:
            return str(i)
    print "No testing fold found for patient", patient_id, "using fold 0"
    return str(0)

ga_file = "/vol/vipdata/data/fetal_data/motion_correction/ga.tsv"

output_folder = "/vol/vipdata/data/fetal_data/OUTPUT/whole_body_shape_padding50"

all_ga = {}
reader = csv.reader( open( ga_file, "rb"), delimiter=" " )
for patient_id, ga in reader:
    all_ga[patient_id] = float(ga)
    
def run_detection( filename ):
    file_id = os.path.basename(filename).split('.')[0]
    if '_' in os.path.basename(filename):
        patient_id = file_id.split('_')[0]
    else:
        patient_id = file_id
    print patient_id 
    if patient_id not in all_ga:
        return

    ga = all_ga[patient_id]
    fold = get_fold(patient_id)
    
    # brain detection
    #SCRIPT_DIR = "/vol/biomedic/users/kpk09/gitlab/irtk/wrapping/cython/scripts/"
    vocabulary = "/vol/medic02/users/kpk09/OUTPUT/brain-detector/model/"+fold+"/vocabulary_"+fold+".npy"
    mser_detector = "/vol/medic02/users/kpk09/OUTPUT/brain-detector/model/"+fold+"/mser_detector_"+fold+"_LinearSVC"
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

    ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"
    output_folder1 = output_folder + "/" + file_id + "/prediction_1/"
    output_folder2 = output_folder + "/" + file_id + "/prediction_2"
    detector1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_pca/2379"
    detector2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_pca/2379"
    shape_model = "/vol/medic02/users/kpk09/OUTPUT/stage1_shape/2379/shape_model.pk"

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

run_detection( sys.argv[1] )
