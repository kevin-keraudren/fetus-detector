#!/usr/bin/python

import irtk
import scipy.ndimage as nd
import numpy as np
from skimage import morphology

left_lung = irtk.imread("left_lung_prior.nii.gz",force_neurological=False)
right_lung = irtk.imread("right_lung_prior.nii.gz",force_neurological=False)
liver = irtk.imread("liver_prior.nii.gz",force_neurological=False)

average = irtk.imread("/vol/medic02/users/kpk09/gitlab/fetus-detector/body-detector/notebooks/tmp/new_average_heart_center.nii.gz",force_neurological=False)

seg = irtk.imread("/vol/medic02/users/kpk09/gitlab/fetus-detector/body-detector/notebooks/tmp/seg_template.nii.gz",force_neurological=False)

res = irtk.zeros(average.get_header(),dtype='int32')

heart_center = np.array(nd.center_of_mass( (seg == 5).view(np.ndarray) ),
                      dtype='float32')
heart = np.argwhere( (seg == 5).view(np.ndarray) ).astype('float32')
r_heart = np.linalg.norm(heart_center-heart,axis=1).mean()

ball = irtk.Image( morphology.ball(r_heart) )
ball.header['origin'] = np.array([0,0,0],dtype='float64')
ball2 = ball.transform(target=liver)
res[ball2>0] = 5

brain_center = np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                      dtype='float32')
brain = np.argwhere( (seg == 2).view(np.ndarray) ).astype('float32')
r_brain = np.linalg.norm(brain_center-brain,axis=1).mean()

ball = irtk.Image( morphology.ball(r_brain) )
ball.header['origin'] = np.array(liver.ImageToWorld(brain_center[::-1]),dtype='float64')
ball2 = ball.transform(target=liver)
res[ball2>0] = 2

threshold = 0.5
res[left_lung>threshold] = 3
res[right_lung>threshold] = 4
res[liver>threshold] = 8

irtk.imwrite("model.nii.gz",res)
