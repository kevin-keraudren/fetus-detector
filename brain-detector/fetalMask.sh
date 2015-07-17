#!/bin/bash

set -x
set -e

filename=$1
ga=$2
output_folder=$3

if [ -z "$output_folder" ];
then
    name=`basename $filename`
    name=(${name//./ })
    name=${name[0]}

    output_folder=$name
fi

mkdir -p ${output_folder}

tmp_mask=$output_folder/mser_`basename $filename`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

classifier=/vol/medic02/users/kpk09/OUTPUT/brain-detector/model/0/mser_detector_0_LinearSVC
vocabulary=/vol/medic02/users/kpk09/OUTPUT/brain-detector/model/0/vocabulary_0.npy

python $DIR/fetalMask_detection.py $filename $ga $tmp_mask \
    --classifier $classifier \
    --vocabulary $vocabulary

python $DIR/fetalMask_segmentation.py --img $filename \
    --ga $ga \
    --mask $tmp_mask \
    --output_dir $output_folder \
    --do_3D \
    --mass \
    -l 20

