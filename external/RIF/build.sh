#!/bin/bash

set -x
set -e

# conda install -c https://conda.binstar.org/richli fftw

export PREFIX=/vol/biomedic/users/kpk09/anaconda/myanaconda
export PYTHON_LIBRARIES="/vol/biomedic/users/kpk09/anaconda/myanaconda/lib/libpython2.7.so"
export PYTHON_INCLUDE_DIRS="/vol/biomedic/users/kpk09/anaconda/myanaconda/include/python2.7;/vol/biomedic/users/kpk09/anaconda/myanaconda/include/python2.7"

mkdir -p build
cd build

cmake -D IRTK_DIR=$PREFIX/lib/                                     \
      -D FFTW3_DIR=$PREFIX/lib/                                   \
      -D FFTW3_LIBRARIES="/vol/biomedic/users/kpk09/anaconda/myanaconda/lib/libfftw3.so;/vol/biomedic/users/kpk09/anaconda/myanaconda/lib/libfftw3f.so;/vol/biomedic/users/kpk09/anaconda/myanaconda/lib/libfftw3f_threads.so;/vol/biomedic/users/kpk09/anaconda/myanaconda/lib/libfftw3_threads.so" \
      -D PNG_LIBRARY:FILEPATH=$PREFIX/lib/libpng.so                    \
      -D TBB_INCLUDE_DIRS:DIRPATH=$PREFIX/include/                     \
      -D TBB_LIBRARY_DIRS:DIRPATH=$PREFIX/lib                          \
      -D PREFIX=$PREFIX \
      -D PYTHON_LIBRARY=$PYTHON_LIBRARIES \
      -D PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIRS \
      ..

make VERBOSE=1
