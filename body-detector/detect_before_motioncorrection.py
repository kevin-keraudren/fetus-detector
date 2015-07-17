#!/usr/bin/python

import os
from glob import glob

import sys
sys.path.append("../commonlib")
from SimpleSlurm import sbatch

import subprocess
from joblib import Parallel, delayed

n_jobs = 10
partition = "interactive"

landmark_folder = "/vol/vipdata/data/fetal_data/motion_correction/landmarks/"
nii_folder = "/vol/vipdata/data/fetal_data/motion_correction/original_scans/"

all_files = sorted(glob(landmark_folder+'/*_landmarks.nii.gz'))

all_cmd = []
for f in all_files:
    file_id = os.path.basename(f)[:-len('_landmarks.nii.gz')]
    filename = nii_folder + "/" + file_id + ".nii"
    
    cmd = [ "python", "detect_all.py", filename ] 
        
    sbatch( cmd,
            mem=32,
            c=n_jobs,
            partition=partition,
            verbose=False,
            dryrun=False )
    #all_cmd.append(cmd)
    
#Parallel(n_jobs=3)(delayed(subprocess.call)(cmd) for cmd in all_cmd )
