#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport cython
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)
        
cdef float integrate( np.ndarray[float, ndim=3,  mode="c"] sat,
                      int d0, int r0, int c0,
                      int d1, int r1, int c1 ):
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of float
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Sum over the given window.
    """

    cdef float S = sat[d1, r1, c1]

    if (d0 - 1 >= 0):
        S -= sat[d0-1, r1, c1]
        
    if (r0 - 1 >= 0):
        S -= sat[d1, r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[d1, r1, c0 - 1]    

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d1, r0 - 1, c0 - 1]

    if (d0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d0 - 1, r1, c0 - 1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0):
        S += sat[d0 - 1, r0 - 1, c1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S -= sat[d0 - 1, r0 - 1, c0 - 1]         

    return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1))

def mean_filter( np.ndarray[float, ndim=3, mode="c"] image, int s0, int s1, int s2 ):
    sat = np.cumsum(np.cumsum(np.cumsum(image,2),1),0)

    filtered = np.zeros( (image.shape[0],
                          image.shape[1],
                          image.shape[2]),
                         dtype=np.float32 )

    cdef int s_0 = s0/2
    cdef int s_1 = s1/2
    cdef int s_2 = s2/2
    cdef int size = s0*s1*s2
    cdef int i, j, k
    cdef int i0, j0, k0
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            for k in xrange(image.shape[2]):
                i0 = i - s_0
                j0 = j - s_1
                k0 = k - s_2
                # subtract 1 because `i_end` and `j_end` are used for indexing into
                # summed-area table, instead of slicing windows of the image.
                i0_end = i0 + s0 - 1
                j0_end = j0 + s1 - 1
                k0_end = k0 + s2 - 1

                i0_end = min(i0_end,sat.shape[0]-1)
                j0_end = min(j0_end,sat.shape[1]-1)
                k0_end = min(k0_end,sat.shape[2]-1)
    
                filtered[i,j,k] = integrate( sat,
                                             i0, j0, k0,
                                             i0_end, j0_end, k0_end)

    return filtered

cdef float patch_mean( np.ndarray[float, ndim=3,  mode="c"] sat,
                       int z, int y, int x,
                       int cz, int cy, int cx,
                       int dz, int dy, int dx ):
        
    cdef int d0 = max(0,z+cz-dz)
    cdef int r0 = max(0,y+cy-dy)
    cdef int c0 = max(0,x+cx-dx)

    d0 = min(d0,sat.shape[0]-1)
    r0 = min(r0,sat.shape[1]-1)
    c0 = min(c0,sat.shape[2]-1)
        
    cdef int d1 = min(z+cz+dz,sat.shape[0]-1)
    cdef int r1 = min(y+cy+dy,sat.shape[1]-1)
    cdef int c1 = min(x+cx+dx,sat.shape[2]-1)

    d1 = max(0,d1)
    r1 = max(0,r1)
    c1 = max(0,c1)

    cdef float S = sat[d1, r1, c1]

    if (d0 - 1 >= 0):
        S -= sat[d0-1, r1, c1]
        
    if (r0 - 1 >= 0):
        S -= sat[d1, r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[d1, r1, c0 - 1]    

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d1, r0 - 1, c0 - 1]

    if (d0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d0 - 1, r1, c0 - 1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0):
        S += sat[d0 - 1, r0 - 1, c1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S -= sat[d0 - 1, r0 - 1, c0 - 1]         

    return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1))
        
    # return integrate( sat,
    #                   d0, r0, c0,
    #                   d1, r1, c1 )

cdef float all_patch_mean( np.ndarray[float, ndim=3,  mode="c"] sat,
                       int z, int y, int x,
                       int cz, int cy, int cx,
                       int dz, int dy, int dx ):
        
    cdef int d0 = max(0,z+cz-dz)
    cdef int r0 = max(0,y+cy-dy)
    cdef int c0 = max(0,x+cx-dx)

    d0 = min(d0,sat.shape[0]-1)
    r0 = min(r0,sat.shape[1]-1)
    c0 = min(c0,sat.shape[2]-1)
        
    cdef int d1 = min(z+cz+dz,sat.shape[0]-1)
    cdef int r1 = min(y+cy+dy,sat.shape[1]-1)
    cdef int c1 = min(x+cx+dx,sat.shape[2]-1)

    d1 = max(0,d1)
    r1 = max(0,r1)
    c1 = max(0,c1)

    cdef float S = sat[d1, r1, c1]

    if (d0 - 1 >= 0):
        S -= sat[d0-1, r1, c1]
        
    if (r0 - 1 >= 0):
        S -= sat[d1, r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[d1, r1, c0 - 1]    

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d1, r0 - 1, c0 - 1]

    if (d0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d0 - 1, r1, c0 - 1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0):
        S += sat[d0 - 1, r0 - 1, c1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S -= sat[d0 - 1, r0 - 1, c0 - 1]         

    return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1))

def get_block_means( np.ndarray[float, ndim=3,  mode="c"] sat,
                     np.ndarray[int, ndim=2,  mode="c"] coords,
                     np.ndarray[int, ndim=2,  mode="c"] offsets,
                     np.ndarray[int, ndim=2,  mode="c"] sizes ):

    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )
    cdef int i, f
    cdef int z, y, x, cz, cy, cx, dz, dy, dx
    cdef int d0, r0, c0, d1, r1, c1
    cdef float S
    cdef int shape0 = sat.shape[0]
    cdef int shape1 = sat.shape[1]
    cdef int shape2 = sat.shape[2]
    for i in xrange(n_coords):
        for f in xrange(n_features):
            # features[i,f] = patch_mean( sat,
            #                             coords[i,0], coords[i,1], coords[i,2],
            #                             offsets[f,0], offsets[f,1], offsets[f,2],
            #                             sizes[f,0], sizes[f,1], sizes[f,2] )
            z = coords[i,0]
            y = coords[i,1]
            x = coords[i,2]
            cz = offsets[f,0]
            cy = offsets[f,1]
            cx = offsets[f,2]
            dz = sizes[f,0]
            dy = sizes[f,1]
            dx = sizes[f,2] 
        
            d0 = max(0,z+cz-dz)
            r0 = max(0,y+cy-dy)
            c0 = max(0,x+cx-dx)

            d0 = min(d0,shape0-1)
            r0 = min(r0,shape1-1)
            c0 = min(c0,shape2-1)
        
            d1 = min(z+cz+dz,shape0-1)
            r1 = min(y+cy+dy,shape1-1)
            c1 = min(x+cx+dx,shape2-1)

            d1 = max(0,d1)
            r1 = max(0,r1)
            c1 = max(0,c1)

            S = sat[d1, r1, c1]

            if (d0 - 1 >= 0):
                S -= sat[d0-1, r1, c1]
        
            if (r0 - 1 >= 0):
                S -= sat[d1, r0 - 1, c1]

            if (c0 - 1 >= 0):
                S -= sat[d1, r1, c0 - 1]    

            if (r0 - 1 >= 0) and (c0 - 1 >= 0):
                S += sat[d1, r0 - 1, c0 - 1]

            if (d0 - 1 >= 0) and (c0 - 1 >= 0):
                S += sat[d0 - 1, r1, c0 - 1]

            if (d0 - 1 >= 0) and (r0 - 1 >= 0):
                S += sat[d0 - 1, r0 - 1, c1]

            if (d0 - 1 >= 0) and (r0 - 1 >= 0) and (c0 - 1 >= 0):
                S -= sat[d0 - 1, r0 - 1, c0 - 1]         

            features[i,f] = S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1))


    return features

def hough_votes( np.ndarray[int, ndim=4,  mode="c"] offsets,
                 np.ndarray[float, ndim=3,  mode="c"] weights=np.ones((0,0,0),dtype='float32'),
                 int padding=0 ):
    cdef int shape0 = offsets.shape[1]
    cdef int shape1 = offsets.shape[2]
    cdef int shape2 = offsets.shape[3]

    if weights.shape[0] == 0:
        weights = np.ones( (shape0,
                            shape1,
                            shape2), dtype='float32')

    cdef np.ndarray[float, ndim=3,  mode="c"] res = np.zeros( (shape0+2*padding,
                                                               shape1+2*padding,
                                                               shape2+2*padding),
                                                              dtype='float32' )

    cdef int x, y, z
    cdef int x0, y0, z0
    cdef double d
    for z in xrange(shape0):
        for y in xrange(shape1):
            for x in xrange(shape2):
                z0 = z + offsets[0,z,y,x]
                y0 = y + offsets[1,z,y,x]
                x0 = x + offsets[2,z,y,x]
                if ( z0 >= -padding and z0 < shape0+padding and
                     y0 >= -padding and y0 < shape1+padding and
                     x0 >= -padding and x0 < shape2+padding ):
                    d = sqrt( offsets[0,z,y,x]**2 +
                              offsets[1,z,y,x]**2 +
                              offsets[2,z,y,x]**2 )
                    res[z0+padding,y0+padding,x0+padding] += weights[z,y,x]

    return res

def crossed_hough_votes( np.ndarray[int, ndim=4,  mode="c"] offsets0,
                         np.ndarray[int, ndim=4,  mode="c"] offsets1,
                         np.ndarray[float, ndim=3,  mode="c"] weights ):
    """
    Uses the votes from offsets1 to look in weights and put back in offsets0.
    """
    cdef int shape0 = offsets0.shape[1]
    cdef int shape1 = offsets0.shape[2]
    cdef int shape2 = offsets0.shape[3]

    cdef np.ndarray[float, ndim=3,  mode="c"] res = np.zeros( (shape0,
                                                               shape1,
                                                               shape2),
                                                              dtype='float32' )

    cdef int x, y, z
    cdef int x0, y0, z0
    cdef int x1, y1, z1
    for z in xrange(shape0):
        for y in xrange(shape1):
            for x in xrange(shape2):
                z0 = z + offsets0[0,z,y,x]
                y0 = y + offsets0[1,z,y,x]
                x0 = x + offsets0[2,z,y,x]
                z1 = z + offsets1[0,z,y,x]
                y1 = y + offsets1[1,z,y,x]
                x1 = x + offsets1[2,z,y,x]
                if ( z0 >= 0 and z0 < shape0 and
                     y0 >= 0 and y0 < shape1 and
                     x0 >= 0 and x0 < shape2 and
                     z1 >= 0 and z1 < shape0 and
                     y1 >= 0 and y1 < shape1 and
                     x1 >= 0 and x1 < shape2 ):
                    res[z0,y0,x0] += weights[z1,y1,x1]

    return res

def hough_votes_backprojection( np.ndarray[int, ndim=4,  mode="c"] offsets,
                                np.ndarray[float, ndim=3,  mode="c"] votes,
                                int padding=0 ):
    cdef int shape0 = offsets.shape[1]
    cdef int shape1 = offsets.shape[2]
    cdef int shape2 = offsets.shape[3]

    cdef np.ndarray[float, ndim=3,  mode="c"] res = np.zeros( (shape0,
                                                               shape1,
                                                               shape2),
                                                              dtype='float32' )

    cdef int x, y, z
    cdef int x0, y0, z0
    for z in xrange(shape0):
        for y in xrange(shape1):
            for x in xrange(shape2):
                z0 = z + offsets[0,z,y,x] + padding
                y0 = y + offsets[1,z,y,x] + padding
                x0 = x + offsets[2,z,y,x] + padding
                if ( z0 >= 0 and z0 < shape0+2*padding and
                     y0 >= 0 and y0 < shape1+2*padding and
                     x0 >= 0 and x0 < shape2+2*padding ):
                    res[z,y,x] = votes[z0,y0,x0]

    return res

cdef extern from "patch_features.h":
    void block_means_TBB( float* sat,
                          int shape0, int shape1, int shape2,
                          int* coords, int n_coords,
                          int* offsets, int* sizes, int n_features,
                          float* features,
                          int n_jobs )
    void block_comparisons_TBB( float* sat,
                                int shape0, int shape1, int shape2,
                                int* coords, int n_coords,
                                int* offsets1, int* sizes1,
                                int* offsets2, int* sizes2,
                                int n_features,
                                float* features,
                                int n_jobs )
    void block_comparisons2_TBB( float* sat1, float* sat2,
                                 int shape0, int shape1, int shape2,
                                 int* coords, int n_coords,
                                 int* offsets1, int* sizes1,
                                 int* offsets2, int* sizes2,
                                 int n_features,
                                 float* features,
                                 int n_jobs )
    void block_comparisons_uvw_TBB( float* sat,
                                    int shape0, int shape1, int shape2,
                                    int* coords, int n_coords,
                                    float* u, float* v, float * w,
                                    int* offsets1, int* sizes1,
                                    int* offsets2, int* sizes2,
                                    int n_features,
                                    unsigned char* features,
                                    int n_jobs )
    void grad_comparisons_uvw_TBB( float* gradZ, float* gradY, float* gradX,
                                   int shape0, int shape1, int shape2,
                                   int* coords, int n_coords,
                                   float* u, float* v, float * w,
                                   int* offsets1,
                                   int* offsets2,
                                   int n_features,
                                   unsigned char* features,
                                   int n_jobs )
    void block_means_uvw_TBB( float* sat,
                              int shape0, int shape1, int shape2,
                              int* coords, int n_coords,
                              float* u, float* v, float * w,
                              int* offsets, int* sizes,
                              int n_features,
                              float* features,
                              int n_jobs )
    void pixel_comparisons_uvw_TBB( float* img,
                                    int shape0, int shape1, int shape2,
                                    int* coords, int n_coords,
                                    float* u, float* v, float * w,
                                    int* offsets1, int* offsets2,
                                    int n_features,
                                    float* features,
                                    int n_jobs )

def get_block_means_cpp( np.ndarray[float, ndim=3,  mode="c"] sat,
                         np.ndarray[int, ndim=2,  mode="c"] coords,
                         np.ndarray[int, ndim=2,  mode="c"] offsets,
                         np.ndarray[int, ndim=2,  mode="c"] sizes,
                         int n_jobs=20 ):
    cdef int shape0 = sat.shape[0]
    cdef int shape1 = sat.shape[1]
    cdef int shape2 = sat.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )

    block_means_TBB( <float*> sat.data,
                      shape0, shape1, shape2,
                      <int*> coords.data, n_coords,
                      <int*> offsets.data, <int*> sizes.data, n_features,
                      <float*> features.data,
                      n_jobs )

    return features

def get_block_comparisons_cpp( np.ndarray[float, ndim=3,  mode="c"] sat,
                               np.ndarray[int, ndim=2,  mode="c"] coords,
                               np.ndarray[int, ndim=2,  mode="c"] offsets1,
                               np.ndarray[int, ndim=2,  mode="c"] sizes1,
                               np.ndarray[int, ndim=2,  mode="c"] offsets2,
                               np.ndarray[int, ndim=2,  mode="c"] sizes2,
                               int n_jobs=20 ):
    cdef int shape0 = sat.shape[0]
    cdef int shape1 = sat.shape[1]
    cdef int shape2 = sat.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )

    block_comparisons_TBB( <float*> sat.data,
                            shape0, shape1, shape2,
                            <int*> coords.data, n_coords,
                            <int*> offsets1.data, <int*> sizes1.data,
                            <int*> offsets2.data, <int*> sizes2.data, n_features,
                            <float*> features.data,
                            n_jobs )

    return features

def get_block_comparisons2_cpp( np.ndarray[float, ndim=3,  mode="c"] sat1,
                                np.ndarray[float, ndim=3,  mode="c"] sat2,
                                np.ndarray[int, ndim=2,  mode="c"] coords,
                                np.ndarray[int, ndim=2,  mode="c"] offsets1,
                                np.ndarray[int, ndim=2,  mode="c"] sizes1,
                                np.ndarray[int, ndim=2,  mode="c"] offsets2,
                                np.ndarray[int, ndim=2,  mode="c"] sizes2,
                                int n_jobs=20 ):
    cdef int shape0 = sat1.shape[0]
    cdef int shape1 = sat1.shape[1]
    cdef int shape2 = sat1.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )

    block_comparisons2_TBB( <float*> sat1.data, <float*> sat2.data,
                            shape0, shape1, shape2,
                            <int*> coords.data, n_coords,
                            <int*> offsets1.data, <int*> sizes1.data,
                            <int*> offsets2.data, <int*> sizes2.data, n_features,
                            <float*> features.data,
                            n_jobs )

    return features

def get_block_comparisons_cpp_uvw( np.ndarray[float, ndim=3,  mode="c"] sat,
                                   np.ndarray[int, ndim=2,  mode="c"] coords,
                                   np.ndarray[float, ndim=2,  mode="c"] u,
                                   np.ndarray[float, ndim=2,  mode="c"] v,
                                   np.ndarray[float, ndim=2,  mode="c"] w,
                                   np.ndarray[int, ndim=2,  mode="c"] offsets1,
                                   np.ndarray[int, ndim=2,  mode="c"] sizes1,
                                   np.ndarray[int, ndim=2,  mode="c"] offsets2,
                                   np.ndarray[int, ndim=2,  mode="c"] sizes2,
                                   int n_jobs=20 ):
    cdef int shape0 = sat.shape[0]
    cdef int shape1 = sat.shape[1]
    cdef int shape2 = sat.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[unsigned char, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                            n_features),
                                                                           dtype='uint8' )

    block_comparisons_uvw_TBB( <float*> sat.data,
                                shape0, shape1, shape2,
                                <int*> coords.data, n_coords,
                                <float*> u.data, <float*> v.data, <float*> w.data,
                                <int*> offsets1.data, <int*> sizes1.data,
                                <int*> offsets2.data, <int*> sizes2.data, n_features,
                                <unsigned char*> features.data,
                                n_jobs )

    return features

def get_grad_comparisons_cpp_uvw( np.ndarray[float, ndim=3,  mode="c"] gradZ,
                                  np.ndarray[float, ndim=3,  mode="c"] gradY,
                                  np.ndarray[float, ndim=3,  mode="c"] gradX,
                                  np.ndarray[int, ndim=2,  mode="c"] coords,
                                  np.ndarray[int, ndim=2,  mode="c"] offsets1,
                                  np.ndarray[int, ndim=2,  mode="c"] offsets2,
                                  np.ndarray[float, ndim=2,  mode="c"] u,
                                  np.ndarray[float, ndim=2,  mode="c"] v,
                                  np.ndarray[float, ndim=2,  mode="c"] w,
                                  int n_jobs=20 ):

    cdef int shape0 = gradZ.shape[0]
    cdef int shape1 = gradZ.shape[1]
    cdef int shape2 = gradZ.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[unsigned char, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                            n_features),
                                                                           dtype='uint8' )
    grad_comparisons_uvw_TBB( <float*> gradZ.data,
                              <float*> gradY.data,
                              <float*> gradX.data,
                              shape0, shape1, shape2,
                              <int*> coords.data, n_coords,
                              <float*> u.data, <float*> v.data, <float*> w.data,
                              <int*> offsets1.data,
                              <int*> offsets2.data,  n_features,
                              <unsigned char*> features.data,
                              n_jobs )

    return features

def get_block_means_cpp_uvw( np.ndarray[float, ndim=3,  mode="c"] sat,
                                   np.ndarray[int, ndim=2,  mode="c"] coords,
                                   np.ndarray[float, ndim=2,  mode="c"] u,
                                   np.ndarray[float, ndim=2,  mode="c"] v,
                                   np.ndarray[float, ndim=2,  mode="c"] w,
                                   np.ndarray[int, ndim=2,  mode="c"] offsets,
                                   np.ndarray[int, ndim=2,  mode="c"] sizes,
                                   int n_jobs=20 ):
    cdef int shape0 = sat.shape[0]
    cdef int shape1 = sat.shape[1]
    cdef int shape2 = sat.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )

    block_means_uvw_TBB( <float*> sat.data,
                          shape0, shape1, shape2,
                          <int*> coords.data, n_coords,
                          <float*> u.data, <float*> v.data, <float*> w.data,
                          <int*> offsets.data, <int*> sizes.data, n_features,
                          <float*> features.data,
                          n_jobs )

    return features

def get_grad_uvw( np.ndarray[int, ndim=2,  mode="c"] coords,
                  np.ndarray[int, ndim=2,  mode="c"] offsets,
                  np.ndarray[float, ndim=3,  mode="c"] gradZ,
                  np.ndarray[float, ndim=3,  mode="c"] gradY,
                  np.ndarray[float, ndim=3,  mode="c"] gradX,
                  np.ndarray[float, ndim=2,  mode="c"] u,
                  np.ndarray[float, ndim=2,  mode="c"] v,
                  np.ndarray[float, ndim=2,  mode="c"] w ):

    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    3*n_features),
                                                                   dtype='float32' )
    cdef int i, f
    cdef int z, y, x
    cdef int shape0 = gradZ.shape[0]
    cdef int shape1 = gradZ.shape[1]
    cdef int shape2 = gradZ.shape[2]
    for i in xrange(n_coords):
        for f in xrange(n_features):
            z = int(coords[i,0] + offsets[f,0]*u[i,0] + offsets[f,1]*v[i,0] + offsets[f,2]*w[i,0])
            y = int(coords[i,1] + offsets[f,0]*u[i,1] + offsets[f,1]*v[i,1] + offsets[f,2]*w[i,1])
            x = int(coords[i,2] + offsets[f,0]*u[i,2] + offsets[f,1]*v[i,2] + offsets[f,2]*w[i,2])
            if ( z >= 0 and z < shape0 and
                 y >= 0 and y < shape1 and
                 x >= 0 and x < shape2 ):
                features[i,3*f+0] = gradZ[z,y,x]*u[i,0]+gradY[z,y,x]*u[i,1]+gradX[z,y,x]*u[i,2]
                features[i,3*f+1] = gradZ[z,y,x]*v[i,0]+gradY[z,y,x]*v[i,1]+gradX[z,y,x]*v[i,2]
                features[i,3*f+2] = gradZ[z,y,x]*w[i,0]+gradY[z,y,x]*w[i,1]+gradX[z,y,x]*w[i,2]
            
    return features

def get_grad_comparisons_uvw( np.ndarray[int, ndim=2,  mode="c"] coords,
                              np.ndarray[int, ndim=2,  mode="c"] offsets1,
                              np.ndarray[int, ndim=2,  mode="c"] offsets2,
                              np.ndarray[float, ndim=3,  mode="c"] gradZ,
                              np.ndarray[float, ndim=3,  mode="c"] gradY,
                              np.ndarray[float, ndim=3,  mode="c"] gradX,
                              np.ndarray[float, ndim=2,  mode="c"] u,
                              np.ndarray[float, ndim=2,  mode="c"] v,
                              np.ndarray[float, ndim=2,  mode="c"] w ):

    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[unsigned char, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                            3*n_features),
                                                                           dtype='uint8' )
    cdef int i, f
    cdef int z1, y1, x1, z2, y2, x2
    cdef int shape0 = gradZ.shape[0]
    cdef int shape1 = gradZ.shape[1]
    cdef int shape2 = gradZ.shape[2]
    for i in xrange(n_coords):
        for f in xrange(n_features):
            z1 = int(coords[i,0] + offsets1[f,0]*u[i,0] + offsets1[f,1]*v[i,0] + offsets1[f,2]*w[i,0])
            y1 = int(coords[i,1] + offsets1[f,0]*u[i,1] + offsets1[f,1]*v[i,1] + offsets1[f,2]*w[i,1])
            x1 = int(coords[i,2] + offsets1[f,0]*u[i,2] + offsets1[f,1]*v[i,2] + offsets1[f,2]*w[i,2])
            z2 = int(coords[i,0] + offsets2[f,0]*u[i,0] + offsets2[f,1]*v[i,0] + offsets2[f,2]*w[i,0])
            y2 = int(coords[i,1] + offsets2[f,0]*u[i,1] + offsets2[f,1]*v[i,1] + offsets2[f,2]*w[i,1])
            x2 = int(coords[i,2] + offsets2[f,0]*u[i,2] + offsets2[f,1]*v[i,2] + offsets2[f,2]*w[i,2])
            if ( z1 >= 0 and z1 < shape0 and
                 y1 >= 0 and y1 < shape1 and
                 x1 >= 0 and x1 < shape2 and
                 z2 >= 0 and z2 < shape0 and
                 y2 >= 0 and y2 < shape1 and
                 x2 >= 0 and x2 < shape2 ):
                features[i,3*f+0] = gradZ[z1,y1,x1]*u[i,0]+gradY[z1,y1,x1]*u[i,1]+gradX[z1,y1,x1]*u[i,2] > gradZ[z2,y2,x2]*u[i,0]+gradY[z2,y2,x2]*u[i,1]+gradX[z2,y2,x2]*u[i,2]
                features[i,3*f+1] = gradZ[z1,y1,x1]*v[i,0]+gradY[z1,y1,x1]*v[i,1]+gradX[z1,y1,x1]*v[i,2] > gradZ[z2,y2,x2]*v[i,0]+gradY[z2,y2,x2]*v[i,1]+gradX[z2,y2,x2]*v[i,2]
                features[i,3*f+2] = gradZ[z1,y1,x1]*w[i,0]+gradY[z1,y1,x1]*w[i,1]+gradX[z1,y1,x1]*w[i,2] > gradZ[z2,y2,x2]*w[i,0]+gradY[z2,y2,x2]*w[i,1]+gradX[z2,y2,x2]*w[i,2]
            
    return features

def get_grad( np.ndarray[int, ndim=2,  mode="c"] coords,
              np.ndarray[int, ndim=2,  mode="c"] offsets,
              np.ndarray[float, ndim=3,  mode="c"] gradZ,
              np.ndarray[float, ndim=3,  mode="c"] gradY,
              np.ndarray[float, ndim=3,  mode="c"] gradX ):

    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    3*n_features),
                                                                   dtype='float32' )
    cdef int i, f
    cdef int z, y, x
    cdef int shape0 = gradZ.shape[0]
    cdef int shape1 = gradZ.shape[1]
    cdef int shape2 = gradZ.shape[2]
    for i in xrange(n_coords):
        for f in xrange(n_features):
            z = coords[i,0] + offsets[f,0]
            y = coords[i,1] + offsets[f,1]
            x = coords[i,2] + offsets[f,2]
            if ( z >= 0 and z < shape0 and
                 y >= 0 and y < shape1 and
                 x >= 0 and x < shape2 ):
                features[i,3*f+0] = gradZ[z,y,x]
                features[i,3*f+1] = gradY[z,y,x]
                features[i,3*f+2] = gradX[z,y,x]
            
    return features

def get_pixel_comparison( np.ndarray[float, ndim=3,  mode="c"] img,
                          np.ndarray[int, ndim=2,  mode="c"] coords,
                          np.ndarray[int, ndim=2,  mode="c"] offsets1,
                          np.ndarray[int, ndim=2,  mode="c"] offsets2 ):

    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )
    cdef int i, f
    cdef int z1, y1, x1, z2, y2, x2
    cdef int shape0 = img.shape[0]
    cdef int shape1 = img.shape[1]
    cdef int shape2 = img.shape[2]
    for i in xrange(n_coords):
        for f in xrange(n_features):
            z1 = coords[i,0] + offsets1[f,0]
            y1 = coords[i,1] + offsets1[f,1]
            x1 = coords[i,2] + offsets1[f,2]
            z2 = coords[i,0] + offsets2[f,0]
            y2 = coords[i,1] + offsets2[f,1]
            x2 = coords[i,2] + offsets2[f,2]
            # boundary condition: reflect
            z1 = min(max(0,z1),shape0-1)
            y1 = min(max(0,y1),shape1-1)
            x1 = min(max(0,x1),shape2-1)
            z2 = min(max(0,z2),shape0-1)
            y2 = min(max(0,y2),shape1-1)
            x2 = min(max(0,x2),shape2-1)
            features[i,f] = img[z1,y1,x1] > img[z2,y2,x2]
            
    return features

def get_pixel_comparisons_cpp_uvw( np.ndarray[float, ndim=3,  mode="c"] img,
                                   np.ndarray[int, ndim=2,  mode="c"] coords,
                                   np.ndarray[float, ndim=2,  mode="c"] u,
                                   np.ndarray[float, ndim=2,  mode="c"] v,
                                   np.ndarray[float, ndim=2,  mode="c"] w,
                                   np.ndarray[int, ndim=2,  mode="c"] offsets1,
                                   np.ndarray[int, ndim=2,  mode="c"] offsets2,
                                   int n_jobs=20 ):
    cdef int shape0 = img.shape[0]
    cdef int shape1 = img.shape[1]
    cdef int shape2 = img.shape[2]
    
    cdef int n_coords = coords.shape[0]
    cdef int n_features = offsets1.shape[0]
    cdef np.ndarray[float, ndim=2,  mode="c"] features = np.zeros( (n_coords,
                                                                    n_features),
                                                                   dtype='float32' )

    pixel_comparisons_uvw_TBB( <float*> img.data,
                                shape0, shape1, shape2,
                                <int*> coords.data, n_coords,
                                <float*> u.data, <float*> v.data, <float*> w.data,
                                <int*> offsets1.data, <int*> offsets2.data,
                                n_features,
                                <float*> features.data,
                                n_jobs )

    return features

def hough_votes27( np.ndarray[int, ndim=4,  mode="c"] offsets,
                   np.ndarray[float, ndim=3,  mode="c"] weights=np.ones((0,0,0),dtype='float32'),
                   double delta=0.0 ):
    cdef int shape0 = offsets.shape[1]
    cdef int shape1 = offsets.shape[2]
    cdef int shape2 = offsets.shape[3]

    if weights.shape[0] == 0:
        weights = np.ones( (shape0,
                            shape1,
                            shape2), dtype='float32')

    cdef np.ndarray[float, ndim=4,  mode="c"] res = np.zeros( (27,
                                                               shape0,
                                                               shape1,
                                                               shape2),
                                                              dtype='float32' )

    cdef int x, y, z
    cdef int x0, y0, z0
    cdef int x27, y27, z27
    cdef double d
    for z in xrange(shape0):
        for y in xrange(shape1):
            for x in xrange(shape2):
                z0 = z + offsets[0,z,y,x]
                y0 = y + offsets[1,z,y,x]
                x0 = x + offsets[2,z,y,x]
                if ( z0 >= 0 and z0 < shape0 and
                     y0 >= 0 and y0 < shape1 and
                     x0 >= 0 and x0 < shape2 ):
                    d = sqrt( offsets[0,z,y,x]**2 +
                              offsets[1,z,y,x]**2 +
                              offsets[2,z,y,x]**2 )
                    if d == 0:
                        res[1 + 3*( 1 + 3*1 ),z0,y0,x0] += weights[z,y,x] * np.log(1.0+d) #exp(-delta*d)
                    else:
                        z27 = round(1.0 + float(offsets[0,z,y,x])/d)
                        y27 = round(1.0 + float(offsets[1,z,y,x])/d)
                        x27 = round(1.0 + float(offsets[2,z,y,x])/d)
                        res[x27 + 3*( y27 + 3*z27 ),z0,y0,x0] += weights[z,y,x] * np.log(1.0+d) #exp(-delta*d)

    return res
