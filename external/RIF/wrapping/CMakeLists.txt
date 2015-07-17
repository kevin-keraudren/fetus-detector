cmake_minimum_required(VERSION 2.6)

# CMakeLists.txt adapted from
# https://github.com/lantiga/cyitk

#project(CYTHON)

find_package(IRTK REQUIRED)
INCLUDE_DIRECTORIES(${IRTK_INCLUDE_DIRS})
LINK_DIRECTORIES(${IRTK_LIBRARIES_DIR})
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# TBB
ADD_DEFINITIONS(-DHAS_TBB)
INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
# LINK_DIRECTORIES(${TBB_LIBRARY_DIRS})
#LINK_LIBRARIES(${TBB_LIBRARIES})

set( IRTK_PYTHON_SRC "/vol/medic02/users/kpk09/github/python-irtk" )
include_directories("${IRTK_PYTHON_SRC}/include/")

find_package(PythonLibs REQUIRED)
if(USE_PYTHONLIBS_FILE) 
  include(${USE_PYTHONLIBS_FILE})
endif(USE_PYTHONLIBS_FILE)

find_package(PythonInterp REQUIRED)
if(USE_PYTHONINTERP_FILE)
  include(${USE_PYTHONINTERP_FILE})
endif(USE_PYTHONINTERP_FILE)

#get numpy include path
execute_process(
  COMMAND
  ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
  OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
  RESULT_VARIABLE NUMPY_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NUMPY_ERR)
  message(SEND_ERROR "WARNING: NumPy header not found.")
endif(NUMPY_ERR)
set(PYTHON_INCLUDE_PATH ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_PATH})

#TODO: Windows? Need a FindCython.cmake
find_program(CYTHON_EXECUTABLE cython /usr/bin /usr/local/bin $ENV{PYTHONPATH} ~/.local/bin)

#create temporary directory
set(TMP_SRC_DIR ${CMAKE_BINARY_DIR}/tmp)
file(MAKE_DIRECTORY ${TMP_SRC_DIR})

set( CPP_SRCS
  "${IRTK_PYTHON_SRC}/src/irtk2cython.cc"
  "rif.cc"
  "${CMAKE_SOURCE_DIR}/image++/src/irtkRotationInvariantFeaturesFilter.cpp")

include_directories(${PYTHON_INCLUDE_PATH})
include_directories("./")
include_directories(${CMAKE_BINARY_DIR}"/tmp")

#run cython on all pyx files
execute_process(
  COMMAND
  ${CYTHON_EXECUTABLE} --cplus
  ${CMAKE_SOURCE_DIR}/wrapping/_rif.pyx -o
  ${CMAKE_BINARY_DIR}/tmp/_rif.cpp
  OUTPUT_VARIABLE CYTHON_OUTPUT
  ERROR_VARIABLE CYTHON_ERROR
  RESULT_VARIABLE CYTHON_ERR
  )
if(CYTHON_ERR)
  message(SEND_ERROR ${CYTHON_ERROR})
endif(CYTHON_ERR)

set(CPP_SRCS ${CPP_SRCS}
  ${CMAKE_BINARY_DIR}/tmp/_rif.cpp)

add_library( "_rif" MODULE
  ${CPP_SRCS})
target_link_libraries( "_rif"
  ${IRTK_LIBRARIES} ${PYTHON_LIBRARY}
  ${IRTK_LIBRARIES} ${FFTW3_LIBRARY} ${FFTW3_LIBRARIES} ${GSL_LIBRARIES}
  ${IRTK_LIBRARIES} ${PNG_LIBRARY} znz)
set_target_properties("_rif"
  PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/wrapping)
                                          
