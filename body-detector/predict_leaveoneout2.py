#!/usr/bin/python

import os
from glob import glob
import irtk
import numpy as np
import scipy.ndimage as nd
import csv

import sys
sys.path.append("../commonlib")
from SimpleSlurm import sbatch

import argparse

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( '-l', type=float, default=0.5 )
parser.add_argument( '--padding', type=int, default=0 )
parser.add_argument( '--chunk_size', type=int, default=int(1e5) )
parser.add_argument( '--patient_id', type=str, default=None )
parser.add_argument( '--n_jobs', type=int, default=5 )
parser.add_argument( '--mem', type=int, default=30 )
parser.add_argument( '--partition', type=str, default='long' )
parser.add_argument( '--iugr', action="store_true", default=False )
parser.add_argument( '--healthy', action="store_true", default=False )
parser.add_argument( '--dryrun', action="store_true", default=False )
parser.add_argument( '--verbose', action="store_true", default=False )
parser.add_argument( '--overwrite', action="store_true", default=False )
parser.add_argument( '--random_offset', type=float, default=0.0 )
args = parser.parse_args()


ground_truth_folder = "/vol/bitbucket/kpk09/detector/data_resampled"

# output_folder1 = "/vol/vipdata/data/fetal_data/OUTPUT/stage1_shape_prediction_padding0/"
# output_folder2 = "/vol/vipdata/data/fetal_data/OUTPUT/stage2_shape_prediction_padding0/"
# detector1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_pca"
# detector2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_pca"

detector1 = "/vol/vipdata/data/fetal_data/OUTPUT/stage1_fast/"
detector2 = "/vol/vipdata/data/fetal_data/OUTPUT/stage2_fast/"
output_folder1 = "/vol/vipdata/data/fetal_data/OUTPUT/stage1_fast_predicton/"
output_folder2 = "/vol/vipdata/data/fetal_data/OUTPUT/stage2_fast_prediction/"

# ## for healthy/iugr
# output_folder1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_shape_prediction_"
# output_folder2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_shape_prediction_"
# detector1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_shape_"
# detector2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_shape_"

if args.healthy:
    output_folder1 += "healthy"
    output_folder2 += "healthy"
    detector1 += "healthy"
    detector2 += "healthy"
if args.iugr:
    # output_folder1 += "iugr"
    # output_folder2 += "iugr"
    # detector1 += "iugr"
    # detector2 += "iugr"
    output_folder1 += "iugrhealthy"
    output_folder2 += "iugrhealthy"
    detector1 = "/vol/medic02/users/kpk09/OUTPUT/stage1_shape_healthy/2381"
    detector2 = "/vol/medic02/users/kpk09/OUTPUT/stage2_shape_healthy/2381"
    shape_model = "/vol/medic02/users/kpk09/OUTPUT/stage1_shape_healthy/2381/shape_model.pk"

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

for f in all_files:
    patient_id = os.path.basename(f)[:-len("_img.nii.gz")]

    if args.iugr and not all_iugr[patient_id]:
        continue
    if args.healthy and all_iugr[patient_id]:
        continue

    if not args.overwrite and os.path.exists( output_folder2 + "/" + patient_id + "/final_seg.nii.gz"):
        continue
    if args.patient_id is not None and patient_id != args.patient_id:
        continue

    seg = irtk.imread(ground_truth_folder+"/"+patient_id+"_seg.nii.gz", force_neurological=True)
    brain_center = np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                            dtype='float32')

    if args.random_offset > 0:
        # add a random offset in pixels to test sensibility to brain localisation
        offset = 2*np.random.rand( n, 3 ) - 1
        offset /= np.linalg.norm(offset)
        offset *= args.random_offset
        brain_center += offset
    
    cmd = [ "python", "predict.py",
            "--input", f ,
            "--output", output_folder1 + "/" + patient_id,
            "--output2", output_folder2 + "/" + patient_id,
            "--detector", detector1 + "/" + patient_id,
            "--detector2", detector2 + "/" + patient_id,
            "--shape_model",  detector1 + "/" + patient_id + "/shape_model.pk",
            "--brain_center"] + map(str,brain_center) + [
            "--chunk_size", str(int(args.chunk_size)),
            "--n_jobs", str(args.n_jobs),
            "-l", str(args.l),
            "--padding", str(args.padding),
            "--shape_optimisation",
            "--narrow_band",
            "--selective",
            "--distribute", str(2),
            "--fast",str(1.5),
            "--theta", str(360)
            ]
        
    sbatch( cmd,
            mem=args.mem,
            c=args.n_jobs,
            partition=args.partition,
            verbose=args.verbose,
            dryrun=args.dryrun )
