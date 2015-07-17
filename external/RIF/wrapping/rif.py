import irtk
import _rif
import numpy as np

def extractRIF(img):
    data = img.rescale(0,1000).get_data('int16','cython')
    header = img.get_header()
    features, header = _rif._extractRIF( data,
                                         header )
    features = np.transpose(features,(3,0,1,2))
    header = img.get_header()
    header['dim'][3] = features.shape[0]
    return irtk.Image(features, header)
