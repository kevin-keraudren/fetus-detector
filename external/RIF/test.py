import irtk
from wrapping import rif

f = "/vol/vipdata/data/fetal_data/motion_correction/original_scans/3879_5.nii"
img = irtk.imread(f, dtype='float32').resample(5.0)

irtk.imwrite( "toto.nii.gz", img )
print img.shape

features = rif.extractRIF(img)

print features.shape

irtk.imwrite("features.nii.gz", features.rescale(0,1000) )

f = "/vol/vipdata/data/fetal_data/motion_correction/original_scans/3879_5.nii"
img = irtk.imread(f, dtype='float32').resample(2.0)

irtk.imwrite( "toto2.nii.gz", img )
print img.shape

features = rif.extractRIF(img)

print features.shape

irtk.imwrite("features2.nii.gz", features[features.shape[0]/2].rescale(0,1000))
