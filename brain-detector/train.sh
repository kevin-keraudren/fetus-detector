#!/bin/bash

# for i in {0..9}; do ./train.sh $i; done

set -x
set -e

n_jobs=10

function mysbatch {
    #echo "$*"
    sbatch --mem=60G -c $n_jobs -p long --wrap="$*"
}

fold=$1

OUTPUT_DIR="/vol/medic02/users/kpk09/OUTPUT/brain-detector/model/$fold"

mkdir -p $OUTPUT_DIR

# input
training_patients="/vol/biomedic/users/kpk09/gitlab/fetal-brain-detection/10_folds/training_$fold.tsv"
data_folder="/vol/vipdata/data/fetal_data/motion_correction/original_scans"
brainboxes="/vol/vipdata/data/fetal_data/motion_correction/brainboxes.tsv"
ga_file="/vol/vipdata/data/fetal_data/motion_correction/ga.tsv"

# output
vocabulary="$OUTPUT_DIR/vocabulary_$fold.npy"
vocabulary_step=2
mser_detector="$OUTPUT_DIR/mser_detector_${fold}_LinearSVC"

    
mysbatch python create_bow.py \
    --training_patients $training_patients \
    --data_folder $data_folder \
    --ga_file $ga_file \
    --step $vocabulary_step \
    --output $vocabulary \
    --n_jobs $n_jobs \
    \&\& python learn_mser.py \
    --training_patients $training_patients \
    --data_folder $data_folder \
    --brainboxes $brainboxes \
    --ga_file $ga_file \
    --vocabulary $vocabulary \
    --n_jobs $n_jobs \
    --output $mser_detector #--debug


