Training a model for localising the fetal brain in MRI
======================================================

1. Learn a vocabulary of 2D SIFT features (extracted using OpenCV) with
MiniBatchKmean from scikit-learn and taking the cluster centers:                    
`create_bow.py`

2. Train an SVM classifier on histogram of SIFT features extractede within MSER regions
(which are first filtered by size using the gestational age):
`learn_mser.py`

Note: these two scripts use SimpleITK instead of IRTK. The main reason I later
switched to IRTK was to correctly handle world coordinates, in particular when
cropping boxes around the brain.


