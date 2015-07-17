#!/usr/bin/python

import os
from glob import glob

import sys
sys.path.insert(1,"../commonlib")
from SimpleSlurm import sbatch

n_jobs = 1
mem = 50
partition = 'long'

detection_folder = "/vol/vipdata/data/fetal_data/OUTPUT/whole_body_shape_padding50/"

all_files = sorted(glob(detection_folder+'/*/prediction_2/img.nii.gz'))

for f in all_files:
    dirname = f[:-len('img.nii.gz')]
    
    cmd = [ "python", "segment_random_walker.py",
            "--img", f,
            "--seg", dirname + "/final_seg.nii.gz",
            "--output", dirname + "/rw10.nii.gz",
            "--narrow_band", str(10),
            "--thorax"
            ] 
        
    sbatch( cmd, mem=mem, c=n_jobs, partition=partition, verbose=False, dryrun=False )
