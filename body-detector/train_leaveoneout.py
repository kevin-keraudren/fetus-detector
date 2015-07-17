#!/usr/bin/python

import os
from glob import glob

import csv

import sys
sys.path.append("../commonlib")
from SimpleSlurm import sbatch

import argparse

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( '--stage', type=int, required=True )
parser.add_argument( '--n_jobs', type=int, default=2 )
parser.add_argument( '--mem', type=int, default=40 )
parser.add_argument( '--partition', type=str, default='long' )
parser.add_argument( '--patient_id', type=str, default=None )
parser.add_argument( '--iugr', action="store_true", default=False )
parser.add_argument( '--healthy', action="store_true", default=False )
parser.add_argument( '--shape_model_only', action="store_true", default=False )
parser.add_argument( '--dryrun', action="store_true", default=False )
parser.add_argument( '--verbose', action="store_true", default=False )
parser.add_argument( '--overwrite', action="store_true", default=False )
args = parser.parse_args()


if args.stage == 1:
    ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"
    #output_folder = "/vol/medic02/users/kpk09/OUTPUT/stage1_pca"
    output_folder = "/vol/vipdata/data/fetal_data/OUTPUT/stage1_fast"
    heart_only = True
    not_centered = False
    factor_background = 1.0
    o_size = 20
    d_size = 15
    stage = ""
    shape_model = True
    n_tests = 100
    n_estimators = 30
    max_depth = 20
    
elif args.stage == 2:
    ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"
    #output_folder = "/vol/medic02/users/kpk09/OUTPUT/stage2_pca"
    output_folder = "/vol/vipdata/data/fetal_data/OUTPUT/stage2_fast"
    heart_only = False
    not_centered = False
    factor_background = 1.0
    o_size = 20
    d_size = 15
    stage = "2"
    shape_model = False
    n_tests = 100
    n_estimators = 30
    max_depth = 20
    
if args.healthy:
    output_folder += "_healthy"
if args.iugr:
    output_folder += "_iugr"

all_iugr = {}
reader = csv.reader( open(
    "/vol/vipdata/data/fetal_data/Mellisa_Damodaram/iugr.tsv", "rb"),
                     delimiter="\t" )
skip = True
for patient_id, iugr, iugr_severity in reader:
    if skip:
        skip = False
        continue
    all_iugr[patient_id] = int(iugr) == 1 

all_files = sorted(glob(ground_truth_folder+'/*_img.nii.gz'))

mysplit = []
for f1 in all_files:
    patient_id1 = os.path.basename(f1).split('_')[0]
    if args.iugr and not all_iugr[patient_id1]:
        continue
    if args.healthy and all_iugr[patient_id1]:
        continue
    train = []
    test = [f1]
    for f2 in all_files:
        patient_id2 = os.path.basename(f2).split('_')[0]
        if f2 != f1:
            if args.iugr and not all_iugr[patient_id2]:
                continue
            if args.healthy and all_iugr[patient_id2]:
                continue
            train.append(f2)

    mysplit.append((train,test))

for train,test in mysplit:
    patient_id = os.path.basename(test[0])[:-len("_img.nii.gz")]

    if args.patient_id is not None and patient_id != args.patient_id:
        continue
    
    if ( not args.overwrite and
         not args.shape_model_only and
         os.path.exists( output_folder + "/" + patient_id + "/reg_heart_01.npy" ) ):
       continue

    seg_files = map(lambda x: x[:-len('_img.nii.gz')]+'_seg.nii.gz', train)
              
    cmd = ( [ "python", "train"+stage+".py",
              "--img" ] + train +
            [ "--seg" ] + seg_files +
            [ "--output_folder", output_folder + "/" + patient_id,
              "--n_jobs", str(args.n_jobs),
              "--factor_background",str(factor_background),
              "--o_size",str(o_size),
              "--d_size",str(d_size),
              "--narrow_band",
              "--n_estimators", str(n_estimators),
              "--n_tests", str(n_tests),
              "--max_depth", str(max_depth) ] )
    if not_centered:
        cmd.append( "--not_centered" )
    if heart_only:
        cmd.append( "--heart_only" )

    if shape_model or args.shape_model_only:
        cmd_shape_model = ( [ "python", "shape_model.py",
                              "--seg" ] + seg_files +
                            [ "--output_folder", output_folder + "/" + patient_id,
                              "--n_jobs", str(args.n_jobs) ] )
        if args.shape_model_only:
            cmd = cmd_shape_model
        else:
            cmd = cmd_shape_model + ["&&"] + cmd
        
    sbatch( cmd,
            mem=args.mem,
            c=args.n_jobs,
            partition=args.partition,
            verbose=args.verbose,
            dryrun=args.dryrun )

