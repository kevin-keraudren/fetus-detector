import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "rif.h":
    void extractRIF( short* img_in,
                          double* pixelSize,
                          double* xAxis,
                          double* yAxis,
                          double* zAxis,
                          double* origin,
                          int* dim,
                          double* img_out,
                          int* dim_out )

def _extractRIF( np.ndarray[short, ndim=4,  mode="c"] img,
                 header ):
    cdef np.ndarray[double, ndim=1,  mode="c"] pixelSize = header['pixelSize']
    cdef np.ndarray[double, ndim=1, mode="c"] xAxis = header['orientation'][0]
    cdef np.ndarray[double, ndim=1, mode="c"] yAxis = header['orientation'][1]
    cdef np.ndarray[double, ndim=1, mode="c"] zAxis = header['orientation'][2]
    cdef np.ndarray[double, ndim=1, mode="c"] origin = header['origin']
    cdef np.ndarray[int, ndim=1, mode="c"] dim =  header['dim']

    # int irtkRotationInvariantFeaturesFilter<VoxelType>::getFeatureDim()
    # _params.kparams.size()*_params.BW+_params.kparams.size();
    cdef int feature_dim = 3*20+3;
    cdef int dim_out[4];
    dim_out[:] = [feature_dim, dim[0], dim[1], dim[2]]
    cdef np.ndarray[double, ndim=4, mode="c"] features = np.zeros( (dim[2],
                                                                    dim[1],
                                                                    dim[0],
                                                                    feature_dim),
                                                                   dtype='float64')

    extractRIF( <short*> img.data,
                <double*> pixelSize.data,
                <double*> xAxis.data,
                <double*> yAxis.data,
                <double*> zAxis.data,
                <double*> origin.data,
                <int*> dim.data,
                <double*> features.data,
                dim_out )

    return features, header
