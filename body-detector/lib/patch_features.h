#ifndef __PATCH_FEATURES_H__
#define __PATCH_FEATURES_H__

#include <algorithm> // max, min, random_shuffle

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

inline size_t index( int y, int x,
                     int shape0, int shape1 ) {
    return x + shape1*y;
}

inline size_t index( int z, int y, int x,
                     int shape0, int shape1, int shape2 ) {
    return x + shape2*( y + shape1*z );
}

inline float mean( float* sat,
                   int shape0, int shape1, int shape2,
                   int z, int y, int x,
                   int cz, int cy, int cx,
                   int dz, int dy, int dx ) {
    
    int d0 = std::max(0,z+cz-dz);
    int r0 = std::max(0,y+cy-dy);
    int c0 = std::max(0,x+cx-dx);

    d0 = std::min(d0,shape0-1);
    r0 = std::min(r0,shape1-1);
    c0 = std::min(c0,shape2-1);
        
    int d1 = std::min(z+cz+dz,shape0-1);
    int r1 = std::min(y+cy+dy,shape1-1);
    int c1 = std::min(x+cx+dx,shape2-1);

    d1 = std::max(0,d1);
    r1 = std::max(0,r1);
    c1 = std::max(0,c1);

    float S = sat[index(d1, r1, c1, shape0, shape1, shape2)];
         
    if (d0-1>=0)
        S -= sat[index(d0-1, r1, c1, shape0, shape1, shape2)];
    if (r0-1>=0)
        S -= sat[index(d1, r0-1, c1, shape0, shape1, shape2)];
    if (c0-1>=0)
        S -= sat[index(d1, r1, c0-1, shape0, shape1, shape2)];
    if (r0-1>=0 && c0-1>=0)
        S += sat[index(d1, r0-1, c0-1, shape0, shape1, shape2)];
    if (d0-1>=0 && c0-1>=0)
        S += sat[index(d0-1, r1, c0-1, shape0, shape1, shape2)];
    if (d0-1>=0 && r0-1>=0)
        S += sat[index(d0-1, r0-1, c1, shape0, shape1, shape2)];
    if (d0-1>=0 && r0-1>=0 && c0-1>=0)
        S -= sat[index(d0-1, r0-1, c0-1, shape0, shape1, shape2)];
        
    return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1));
}

inline float mean( float* sat,
                   int shape0, int shape1, int shape2,
                   int z, int y, int x,
                   int dz, int dy, int dx ) {
    
    int d0 = std::max(0,z-dz);
    int r0 = std::max(0,y-dy);
    int c0 = std::max(0,x-dx);

    d0 = std::min(d0,shape0-1);
    r0 = std::min(r0,shape1-1);
    c0 = std::min(c0,shape2-1);
        
    int d1 = std::min(z+dz,shape0-1);
    int r1 = std::min(y+dy,shape1-1);
    int c1 = std::min(x+dx,shape2-1);

    d1 = std::max(0,d1);
    r1 = std::max(0,r1);
    c1 = std::max(0,c1);

    float S = sat[index(d1, r1, c1, shape0, shape1, shape2)];
         
    if (d0-1>=0)
        S -= sat[index(d0-1, r1, c1, shape0, shape1, shape2)];
    if (r0-1>=0)
        S -= sat[index(d1, r0-1, c1, shape0, shape1, shape2)];
    if (c0-1>=0)
        S -= sat[index(d1, r1, c0-1, shape0, shape1, shape2)];
    if (r0-1>=0 && c0-1>=0)
        S += sat[index(d1, r0-1, c0-1, shape0, shape1, shape2)];
    if (d0-1>=0 && c0-1>=0)
        S += sat[index(d0-1, r1, c0-1, shape0, shape1, shape2)];
    if (d0-1>=0 && r0-1>=0)
        S += sat[index(d0-1, r0-1, c1, shape0, shape1, shape2)];
    if (d0-1>=0 && r0-1>=0 && c0-1>=0)
        S -= sat[index(d0-1, r0-1, c0-1, shape0, shape1, shape2)];
        
    return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1));
}

inline void block_means( float* sat,
                         int shape0, int shape1, int shape2,
                         int* coords, int n_coords,
                         int* offsets, int* sizes, int n_features,
                         float* features ) {
    for ( int i = 0; i < n_coords; i++ )
        for ( int f = 0;  f < n_features; f++ )
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             coords[index(i,0,n_coords,3)],
                                                             coords[index(i,1,n_coords,3)],
                                                             coords[index(i,2,n_coords,3)],
                                                             offsets[index(f,0,n_features,3)],
                                                             offsets[index(f,1,n_features,3)],
                                                             offsets[index(f,2,n_features,3)],
                                                             sizes[index(f,0,n_features,3)],
                                                             sizes[index(f,1,n_features,3)],
                                                             sizes[index(f,2,n_features,3)] );
}

inline void block_comparisons( float* sat,
                               int shape0, int shape1, int shape2,
                               int* coords, int n_coords,
                               int* offsets1, int* sizes1,
                               int* offsets2, int* sizes2,
                               int n_features,
                               float* features ) {

    for ( int i = 0; i < n_coords; i++ )
        for ( int f = 0;  f < n_features; f++ )
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             coords[index(i,0,n_coords,3)],
                                                             coords[index(i,1,n_coords,3)],
                                                             coords[index(i,2,n_coords,3)],
                                                             offsets1[index(f,0,n_features,3)],
                                                             offsets1[index(f,1,n_features,3)],
                                                             offsets1[index(f,2,n_features,3)],
                                                             sizes1[index(f,0,n_features,3)],
                                                             sizes1[index(f,1,n_features,3)],
                                                             sizes1[index(f,2,n_features,3)] ) > mean( sat,
                                                                                                       shape0, shape1, shape2,
                                                                                                       coords[index(i,0,n_coords,3)],
                                                                                                       coords[index(i,1,n_coords,3)],
                                                                                                       coords[index(i,2,n_coords,3)],
                                                                                                       offsets2[index(f,0,n_features,3)],
                                                                                                       offsets2[index(f,1,n_features,3)],
                                                                                                       offsets2[index(f,2,n_features,3)],
                                                                                                       sizes2[index(f,0,n_features,3)],
                                                                                                       sizes2[index(f,1,n_features,3)],
                                                                                                       sizes2[index(f,2,n_features,3)] );
}
class FeaturesTBB_means {
    float* sat;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets;
    int* sizes;
    int n_features;
    float* features;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             coords[index(i,0,n_coords,3)],
                                                             coords[index(i,1,n_coords,3)],
                                                             coords[index(i,2,n_coords,3)],
                                                             offsets[index(f,0,n_features,3)],
                                                             offsets[index(f,1,n_features,3)],
                                                             offsets[index(f,2,n_features,3)],
                                                             sizes[index(f,0,n_features,3)],
                                                             sizes[index(f,1,n_features,3)],
                                                             sizes[index(f,2,n_features,3)] );

      
        }
    }
  }
 FeaturesTBB_means( float* _sat,
              int _shape0, int _shape1, int _shape2,
              int* _coords, int _n_coords,
              int* _offsets, int* _sizes,
              int _n_features,
              float* _features ) {
     sat = _sat;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets = _offsets;
     sizes = _sizes;
     n_features = _n_features;
     features = _features;
 }
};

inline void block_means_TBB( float* sat,
                             int shape0, int shape1, int shape2,
                             int* coords, int n_coords,
                             int* offsets, int* sizes,
                             int n_features,
                             float* features,
                             int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                              FeaturesTBB_means(  sat,
                                            shape0, shape1, shape2,
                                            coords, n_coords,
                                            offsets, sizes,
                                            n_features,
                                            features )
                              );
    init.terminate();
}

class FeaturesTBB_comparisons {
    float* sat;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets1;
    int* sizes1;
    int* offsets2;
    int* sizes2;
    int n_features;
    float* features;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             coords[index(i,0,n_coords,3)],
                                                             coords[index(i,1,n_coords,3)],
                                                             coords[index(i,2,n_coords,3)],
                                                             offsets1[index(f,0,n_features,3)],
                                                             offsets1[index(f,1,n_features,3)],
                                                             offsets1[index(f,2,n_features,3)],
                                                             sizes1[index(f,0,n_features,3)],
                                                             sizes1[index(f,1,n_features,3)],
                                                             sizes1[index(f,2,n_features,3)] ) > mean( sat,
                                                                                                       shape0, shape1, shape2,
                                                                                                       coords[index(i,0,n_coords,3)],
                                                                                                       coords[index(i,1,n_coords,3)],
                                                                                                       coords[index(i,2,n_coords,3)],
                                                                                                       offsets2[index(f,0,n_features,3)],
                                                                                                       offsets2[index(f,1,n_features,3)],
                                                                                                       offsets2[index(f,2,n_features,3)],
                                                                                                       sizes2[index(f,0,n_features,3)],
                                                                                                       sizes2[index(f,1,n_features,3)],
                                                                                                       sizes2[index(f,2,n_features,3)] );

      
        }
    }
  }
 FeaturesTBB_comparisons( float* _sat,
              int _shape0, int _shape1, int _shape2,
              int* _coords, int _n_coords,
              int* _offsets1, int* _sizes1,
              int* _offsets2, int* _sizes2,
              int _n_features,
              float* _features ) {
     sat = _sat;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets1 = _offsets1;
     sizes1 = _sizes1;
     offsets2 = _offsets2;
     sizes2 = _sizes2;
     n_features = _n_features;
     features = _features;
 }
};

inline void block_comparisons_TBB( float* sat,
                                   int shape0, int shape1, int shape2,
                                   int* coords, int n_coords,
                                   int* offsets1, int* sizes1,
                                   int* offsets2, int* sizes2,
                                   int n_features,
                                   float* features,
                                   int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                              FeaturesTBB_comparisons(  sat,
                                            shape0, shape1, shape2,
                                            coords, n_coords,
                                            offsets1, sizes1,
                                            offsets2, sizes2,
                                            n_features,
                                            features )
                              );
    init.terminate();
}

class FeaturesTBB_comparisons2 {
    float* sat1;
    float* sat2;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets1;
    int* sizes1;
    int* offsets2;
    int* sizes2;
    int n_features;
    float* features;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            features[index(i,f,n_coords,n_features)] = mean( sat1,
                                                             shape0, shape1, shape2,
                                                             coords[index(i,0,n_coords,3)],
                                                             coords[index(i,1,n_coords,3)],
                                                             coords[index(i,2,n_coords,3)],
                                                             offsets1[index(f,0,n_features,3)],
                                                             offsets1[index(f,1,n_features,3)],
                                                             offsets1[index(f,2,n_features,3)],
                                                             sizes1[index(f,0,n_features,3)],
                                                             sizes1[index(f,1,n_features,3)],
                                                             sizes1[index(f,2,n_features,3)] ) > mean( sat2,
                                                                                                       shape0, shape1, shape2,
                                                                                                       coords[index(i,0,n_coords,3)],
                                                                                                       coords[index(i,1,n_coords,3)],
                                                                                                       coords[index(i,2,n_coords,3)],
                                                                                                       offsets2[index(f,0,n_features,3)],
                                                                                                       offsets2[index(f,1,n_features,3)],
                                                                                                       offsets2[index(f,2,n_features,3)],
                                                                                                       sizes2[index(f,0,n_features,3)],
                                                                                                       sizes2[index(f,1,n_features,3)],
                                                                                                       sizes2[index(f,2,n_features,3)] );

      
        }
    }
  }
 FeaturesTBB_comparisons2( float* _sat1, float* _sat2,
                           int _shape0, int _shape1, int _shape2,
                           int* _coords, int _n_coords,
                           int* _offsets1, int* _sizes1,
                           int* _offsets2, int* _sizes2,
                           int _n_features,
                           float* _features ) {
     sat1 = _sat1;
     sat2 = _sat2;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets1 = _offsets1;
     sizes1 = _sizes1;
     offsets2 = _offsets2;
     sizes2 = _sizes2;
     n_features = _n_features;
     features = _features;
 }
};

inline void block_comparisons2_TBB( float* sat1,
                                    float* sat2,
                                   int shape0, int shape1, int shape2,
                                   int* coords, int n_coords,
                                   int* offsets1, int* sizes1,
                                   int* offsets2, int* sizes2,
                                   int n_features,
                                   float* features,
                                   int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                      FeaturesTBB_comparisons2(  sat1, sat2,
                                                 shape0, shape1, shape2,
                                                 coords, n_coords,
                                                 offsets1, sizes1,
                                                 offsets2, sizes2,
                                                 n_features,
                                                 features )
                              );
    init.terminate();
}

class FeaturesTBB_comparisons_uvw {
    float* sat;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets1;
    int* sizes1;
    int* offsets2;
    int* sizes2;
    int n_features;
    unsigned char* features;
    float* u;
    float* v;
    float* w;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            int A0 = index(f,0,n_features,3);
            int A1 = index(f,1,n_features,3);
            int A2 = index(f,2,n_features,3);
            int B0 = index(i,0,n_coords,3);
            int B1 = index(i,1,n_coords,3);
            int B2 = index(i,2,n_coords,3);
            int z1 = ( coords[B0] +
                       offsets1[A0]*u[B0] +
                       offsets1[A1]*v[B0] +
                       offsets1[A2]*w[B0] );
            int y1 = ( coords[B1] +
                       offsets1[A0]*u[B1] +
                       offsets1[A1]*v[B1] +
                       offsets1[A2]*w[B1] );
            int x1 = ( coords[B2] +
                       offsets1[A0]*u[B2] +
                       offsets1[A1]*v[B2] +
                       offsets1[A2]*w[B2] );
            int z2 = ( coords[B0] +
                       offsets2[A0]*u[B0] +
                       offsets2[A1]*v[B0] +
                       offsets2[A2]*w[B0] );
            int y2 = ( coords[B1] +
                       offsets2[A0]*u[B1] +
                       offsets2[A1]*v[B1] +
                       offsets2[A2]*w[B1] );
            int x2 = ( coords[B2] +
                       offsets2[A0]*u[B2] +
                       offsets2[A1]*v[B2] +
                       offsets2[A2]*w[B2] );
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             z1, y1, x1,
                                                             sizes1[A0],
                                                             sizes1[A1],
                                                             sizes1[A2] ) > mean( sat,
                                                                                  shape0, shape1, shape2,
                                                                                  z2, y2, x2,
                                                                                  sizes2[A0],
                                                                                  sizes2[A1],
                                                                                  sizes2[A2] );
        }
    }
  }
 FeaturesTBB_comparisons_uvw( float* _sat,
                              int _shape0, int _shape1, int _shape2,
                              int* _coords, int _n_coords,
                              float* _u, float* _v, float* _w,
                              int* _offsets1, int* _sizes1,
                              int* _offsets2, int* _sizes2,
                              int _n_features,
                              unsigned char* _features ) {
     sat = _sat;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets1 = _offsets1;
     sizes1 = _sizes1;
     offsets2 = _offsets2;
     sizes2 = _sizes2;
     n_features = _n_features;
     features = _features;
     u = _u;
     v = _v;
     w = _w;
 }
};

inline void block_comparisons_uvw_TBB( float* sat,
                                       int shape0, int shape1, int shape2,
                                       int* coords, int n_coords,
                                       float* u, float* v, float * w,
                                       int* offsets1, int* sizes1,
                                       int* offsets2, int* sizes2,
                                       int n_features,
                                       unsigned char* features,
                                       int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                              FeaturesTBB_comparisons_uvw(  sat,
                                                            shape0, shape1, shape2,
                                                            coords, n_coords,
                                                            u, v, w,
                                                            offsets1, sizes1,
                                                            offsets2, sizes2,
                                                            n_features,
                                                            features )
                              );
    init.terminate();
}

class FeaturesTBB_grad_comparisons_uvw {
    float* gradZ;
    float* gradY;
    float* gradX;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets1;
    int* offsets2;
    int n_features;
    unsigned char* features;
    float* u;
    float* v;
    float* w;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            int A0 = index(f,0,n_features,3);
            int A1 = index(f,1,n_features,3);
            int A2 = index(f,2,n_features,3);
            int B0 = index(i,0,n_coords,3);
            int B1 = index(i,1,n_coords,3);
            int B2 = index(i,2,n_coords,3);
            int z1 = ( coords[B0] +
                       offsets1[A0]*u[B0] +
                       offsets1[A1]*v[B0] +
                       offsets1[A2]*w[B0] );
            int y1 = ( coords[B1] +
                       offsets1[A0]*u[B1] +
                       offsets1[A1]*v[B1] +
                       offsets1[A2]*w[B1] );
            int x1 = ( coords[B2] +
                       offsets1[A0]*u[B2] +
                       offsets1[A1]*v[B2] +
                       offsets1[A2]*w[B2] );
            int z2 = ( coords[B0] +
                       offsets2[A0]*u[B0] +
                       offsets2[A1]*v[B0] +
                       offsets2[A2]*w[B0] );
            int y2 = ( coords[B1] +
                       offsets2[A0]*u[B1] +
                       offsets2[A1]*v[B1] +
                       offsets2[A2]*w[B1] );
            int x2 = ( coords[B2] +
                       offsets2[A0]*u[B2] +
                       offsets2[A1]*v[B2] +
                       offsets2[A2]*w[B2] );
            if ( z1 >= 0 && z1 < shape0 &&
                 y1 >= 0 && y1 < shape1 &&
                 x1 >= 0 && x1 < shape2 &&
                 z2 >= 0 && z2 < shape0 &&
                 y2 >= 0 && y2 < shape1 &&
                 x2 >= 0 && x2 < shape2 ) 
                features[index(i,f,n_coords,n_features)] = gradZ[index(z1,y1,x1,shape0,shape1,shape2)]*gradZ[index(z2,y2,x2,shape0,shape1,shape2)] +
                                                           gradY[index(z1,y1,x1,shape0,shape1,shape2)]*gradY[index(z2,y2,x2,shape0,shape1,shape2)] +
                                                           gradX[index(z1,y1,x1,shape0,shape1,shape2)]*gradX[index(z2,y2,x2,shape0,shape1,shape2)]  > 0;
        }
    }
  }
 FeaturesTBB_grad_comparisons_uvw( float* _gradZ, float* _gradY, float* _gradX,
                              int _shape0, int _shape1, int _shape2,
                              int* _coords, int _n_coords,
                              float* _u, float* _v, float* _w,
                              int* _offsets1,
                              int* _offsets2,
                              int _n_features,
                              unsigned char* _features ) {
     gradZ = _gradZ;
     gradY = _gradY;
     gradX = _gradX;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets1 = _offsets1;
     offsets2 = _offsets2;
     n_features = _n_features;
     features = _features;
     u = _u;
     v = _v;
     w = _w;
 }
};

inline void grad_comparisons_uvw_TBB( float* gradZ, float* gradY, float* gradX,
                                      int shape0, int shape1, int shape2,
                                      int* coords, int n_coords,
                                      float* u, float* v, float * w,
                                      int* offsets1,
                                      int* offsets2,
                                      int n_features,
                                      unsigned char* features,
                                      int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                      FeaturesTBB_grad_comparisons_uvw(  gradZ, gradY, gradX,
                                                         shape0, shape1, shape2,
                                                         coords, n_coords,
                                                         u, v, w,
                                                         offsets1, 
                                                         offsets2, 
                                                         n_features,
                                                         features )
                              );
    init.terminate();
}

class FeaturesTBB_means_uvw {
    float* sat;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets;
    int* sizes;
    int n_features;
    float* features;
    float* u;
    float* v;
    float* w;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            int z = ( coords[index(i,0,n_coords,3)] +
                       offsets[index(f,0,n_features,3)]*u[index(i,0,n_features,3)] +
                       offsets[index(f,1,n_features,3)]*v[index(i,0,n_features,3)] +
                       offsets[index(f,2,n_features,3)]*w[index(i,0,n_features,3)] );
            int y = ( coords[index(i,1,n_coords,3)] +
                       offsets[index(f,0,n_features,3)]*u[index(i,1,n_features,3)] +
                       offsets[index(f,1,n_features,3)]*v[index(i,1,n_features,3)] +
                       offsets[index(f,2,n_features,3)]*w[index(i,1,n_features,3)] );
            int x = ( coords[index(i,2,n_coords,3)] +
                       offsets[index(f,0,n_features,3)]*u[index(i,2,n_features,3)] +
                       offsets[index(f,1,n_features,3)]*v[index(i,2,n_features,3)] +
                       offsets[index(f,2,n_features,3)]*w[index(i,2,n_features,3)] );
            features[index(i,f,n_coords,n_features)] = mean( sat,
                                                             shape0, shape1, shape2,
                                                             z, y, x,
                                                             sizes[index(f,0,n_features,3)],
                                                             sizes[index(f,1,n_features,3)],
                                                             sizes[index(f,2,n_features,3)] );

      
        }
    }
  }
 FeaturesTBB_means_uvw( float* _sat,
                              int _shape0, int _shape1, int _shape2,
                              int* _coords, int _n_coords,
                              float* _u, float* _v, float* _w,
                              int* _offsets, int* _sizes,
                              int _n_features,
                              float* _features ) {
     sat = _sat;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets = _offsets;
     sizes = _sizes;
     n_features = _n_features;
     features = _features;
     u = _u;
     v = _v;
     w = _w;
 }
};

inline void block_means_uvw_TBB( float* sat,
                                       int shape0, int shape1, int shape2,
                                       int* coords, int n_coords,
                                       float* u, float* v, float * w,
                                       int* offsets, int* sizes,
                                       int n_features,
                                       float* features,
                                       int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                              FeaturesTBB_means_uvw(  sat,
                                                            shape0, shape1, shape2,
                                                            coords, n_coords,
                                                            u, v, w,
                                                            offsets, sizes,
                                                            n_features,
                                                            features )
                              );
    init.terminate();
}

class FeaturesTBB_pixel_comparisons_uvw {
    float* img;
    int shape0, shape1, shape2;
    int* coords;
    int n_coords;
    int* offsets1;
    int* offsets2;
    int n_features;
    float* features;
    float* u;
    float* v;
    float* w;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t f = r.begin(); f != r.end(); ++f ) {
        for ( int i = 0; i < n_coords; i++ ) {
            int z1 = ( coords[index(i,0,n_coords,3)] +
                       offsets1[index(f,0,n_features,3)]*u[index(i,0,n_features,3)] +
                       offsets1[index(f,1,n_features,3)]*v[index(i,0,n_features,3)] +
                       offsets1[index(f,2,n_features,3)]*w[index(i,0,n_features,3)] );
            int y1 = ( coords[index(i,1,n_coords,3)] +
                       offsets1[index(f,0,n_features,3)]*u[index(i,1,n_features,3)] +
                       offsets1[index(f,1,n_features,3)]*v[index(i,1,n_features,3)] +
                       offsets1[index(f,2,n_features,3)]*w[index(i,1,n_features,3)] );
            int x1 = ( coords[index(i,2,n_coords,3)] +
                       offsets1[index(f,0,n_features,3)]*u[index(i,2,n_features,3)] +
                       offsets1[index(f,1,n_features,3)]*v[index(i,2,n_features,3)] +
                       offsets1[index(f,2,n_features,3)]*w[index(i,2,n_features,3)] );
            int z2 = ( coords[index(i,0,n_coords,3)] +
                       offsets2[index(f,0,n_features,3)]*u[index(i,0,n_features,3)] +
                       offsets2[index(f,1,n_features,3)]*v[index(i,0,n_features,3)] +
                       offsets2[index(f,2,n_features,3)]*w[index(i,0,n_features,3)] );
            int y2 = ( coords[index(i,1,n_coords,3)] +
                       offsets2[index(f,0,n_features,3)]*u[index(i,1,n_features,3)] +
                       offsets2[index(f,1,n_features,3)]*v[index(i,1,n_features,3)] +
                       offsets2[index(f,2,n_features,3)]*w[index(i,1,n_features,3)] );
            int x2 = ( coords[index(i,2,n_coords,3)] +
                       offsets2[index(f,0,n_features,3)]*u[index(i,2,n_features,3)] +
                       offsets2[index(f,1,n_features,3)]*v[index(i,2,n_features,3)] +
                       offsets2[index(f,2,n_features,3)]*w[index(i,2,n_features,3)] );

            // boundary condition: reflect
            z1 = std::min( std::max(0,z1), shape0-1);
            y1 = std::min( std::max(0,y1), shape1-1);
            x1 = std::min( std::max(0,x1), shape2-1);
            
            z2 = std::min( std::max(0,z2), shape0-1);
            y2 = std::min( std::max(0,y2), shape1-1);
            x2 = std::min( std::max(0,x2), shape2-1);
        
            features[index(i,f,n_coords,n_features)] = img[index(z1, y1, x1,
                                                                 shape0, shape1, shape2)] > img[index(z2, y2, x2,
                                                                                                      shape0, shape1, shape2)];
      
        }
    }
  }
 FeaturesTBB_pixel_comparisons_uvw( float* _img,
                                    int _shape0, int _shape1, int _shape2,
                                    int* _coords, int _n_coords,
                                    float* _u, float* _v, float* _w,
                                    int* _offsets1, int* _offsets2,
                                    int _n_features,
                                    float* _features ) {
     img = _img;
     shape0 = _shape0;
     shape1 = _shape1;
     shape2 = _shape2;
     coords = _coords;
     n_coords = _n_coords;
     offsets1 = _offsets1;
     offsets2 = _offsets2;
     n_features = _n_features;
     features = _features;
     u = _u;
     v = _v;
     w = _w;
 }
};

inline void pixel_comparisons_uvw_TBB( float* img,
                                       int shape0, int shape1, int shape2,
                                       int* coords, int n_coords,
                                       float* u, float* v, float * w,
                                       int* offsets1, int* offsets2, 
                                       int n_features,
                                       float* features,
                                       int n_jobs ) {

    tbb::task_scheduler_init init( n_jobs );
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_features ),
                      FeaturesTBB_pixel_comparisons_uvw( img,
                                                         shape0, shape1, shape2,
                                                         coords, n_coords,
                                                         u, v, w,
                                                         offsets1, offsets2,
                                                         n_features,
                                                         features )
                      );
    init.terminate();
}


#endif // __PATCH_FEATURES_H__
