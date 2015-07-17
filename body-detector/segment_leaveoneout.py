#!/usr/bin/python

import os
from glob import glob

import sys
sys.path.insert(1,"../commonlib")
from SimpleSlurm import sbatch

n_jobs = 5
mem = 30
partition = 'long'

ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"
output_folder1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_pca_prediction"
output_folder2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_pca_prediction"
detector1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_pca"
detector2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_pca"

all_files = sorted(glob(ground_truth_folder+'/*_img.nii.gz'))

for f in all_files:
    patient_id = os.path.basename(f)[:-len("_img.nii.gz")]

    #if os.path.exists( output_folder2 + "/" + patient_id + "/votes_heart.nii.gz"):
    #    continue
    
    cmd = [ "python", "segment_random_walker.py",
            "--img", f,
            "--seg", output_folder2 + "/" + patient_id + "/final_seg.nii.gz",
            "--output", output_folder2 + "/" + patient_id + "/rw.nii.gz",
            ] 
        
    sbatch( cmd, mem=mem, c=n_jobs, partition=partition, verbose=False, dryrun=False )
