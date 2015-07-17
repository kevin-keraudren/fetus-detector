#!/usr/bin/python

import os
from glob import glob

import sys
sys.path.append("lib")
from SimpleSlurm import sbatch

n_jobs = 5

ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"
output_folder = "/vol/medic02/users/kpk09/OUTPUT/stage1_prediction"
heart_only = True
detector = "/vol/medic02/users/kpk09/OUTPUT/stage1"

all_files = sorted(glob(ground_truth_folder+'/*_img.nii.gz'))

for f in all_files:
    patient_id = os.path.basename(f)[:-len("_img.nii.gz")]

    if os.path.exists( output_folder + "/" + patient_id + "/votes_heart.nii.gz"):
        continue
    
    cmd = [ "python", "predict.py",
            "--input", f ,
            "--output", output_folder + "/" + patient_id,
            "--detector", detector + "/" + patient_id,
            "--chunk_size", str(int(1e5)),
            "--n_jobs", str(n_jobs),
            "--narrow_band"
            ] 
    if heart_only:
        cmd.append( "--heart_only" )
        
    sbatch( cmd, mem=30, c=n_jobs, verbose=False, dryrun=True )
