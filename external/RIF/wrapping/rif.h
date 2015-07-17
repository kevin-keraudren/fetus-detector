#include <irtkImage.h>
#include <limits>

#include <irtkResampling.h>

#include <irtkRotationInvariantFeaturesFilter.h>

#include "irtk2cython.h"

void extractRIF( short* img_in,
                 double* pixelSize,
                 double* xAxis,
                 double* yAxis,
                 double* zAxis,
                 double* origin,
                 int* dim,
                 double* img_out,
                 int* dim_out );
