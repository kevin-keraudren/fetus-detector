project(irtkRotationInvariantFilter)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0010 OLD)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (WIN32)
  # for msvc 8 deprecated warning removal
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)
  add_definitions(-DWIN32 -D_WIN32)
  add_definitions(-DNOMINMAX)

  #set(CMAKE_DEBUG_POSTFIX "D")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib;libcmtd.lib")
endif(WIN32)

if (WIN32)
  add_definitions(-bigobj)
endif()

# we do not need to search for IRTK for in source tree build
find_package(IRTK REQUIRED)
message("${IRTK_LIBRARIES_DIR}")
include_directories(${IRTK_INCLUDE_DIRS})

find_package(FFTW3 REQUIRED)
add_definitions(-D_STA_LINK_FFTW)
include_directories(${FFTW3_INCLUDE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/image++/src/skibbe_base)

add_definitions(-D_STA_LINK_FFTW -D_STA_MULTI_THREAD -D_STA_FFT_MULTI_THREAD)

include_directories(image++/include)

link_directories(${IRTK_LIBRARIES_DIR})

set(RIF_FILTER_HDR image++/include/irtkRotationInvariantFeaturesFilter.h)
set(RIF_FILTER_SRCS image++/src/irtkRotationInvariantFeaturesFilter.cpp)

add_executable(irtkRotationInvariantFilter ${RIF_FILTER_HDR} ${RIF_FILTER_SRCS}
			applications/irtkRotationInvariantFilter.cpp)
target_link_libraries(irtkRotationInvariantFilter ${IRTK_LIBRARIES} ${FFTW3_LIBRARY} ${FFTW3_LIBRARIES} ${GSL_LIBRARIES} ${PNG_LIBRARY}) 

SUBDIRS(wrapping)

