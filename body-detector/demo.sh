#!/bin/bash

set -x
set -e

if [ ! -f stack-1.nii.gz ]; then
	wget https://github.com/kevin-keraudren/example-motion-correction/raw/master/data/stack-1.nii.gz
fi

if [ ! -d ../brain-detector/trained_model ]; then
	cd ../brain-detector/ && unzip trained_model.zip && cd -
fi
if [ ! -d trained_model/stage1 ]; then
	cd trained_model && unzip stage1.zip && cd -
fi
for f in clf reg_heart reg_liver reg_left_lung reg_right_lung
do
	if [ ! -f "trained_model/stage2/$f" ]; then
		cd trained_model/stage2 && unzip $f.zip && cd -
	fi
done

python detect_all.py --filename stack-1.nii.gz --ga 30