# Try to find gnu scientific library GSL
# See 
# http://www.gnu.org/software/gsl/  and 
# http://gnuwin32.sourceforge.net/packages/gsl.htm
#
# Once run this will define: 
# 
# GSL_FOUND       = system has GSL lib
#
# GSL_LIBRARIES   = full path to the libraries
#    on Unix/Linux with additional linker flags from "gsl-config --libs"
# 
# CMAKE_GSL_C_FLAGS  = Unix compiler flags for GSL, essentially "`gsl-config --cflags`"
#
# GSL_INCLUDE_DIR      = where to find headers 
#
# GSL_LINK_DIRECTORIES = link directories, useful for rpath on Unix
# GSL_EXE_LINKER_FLAGS = rpath on Unix
#
# Felix Woelk 07/2004
# Jan Woetzel
#
# www.mip.informatik.uni-kiel.de
# --------------------------------

IF(WIN32)
  # JW tested with gsl-1.8, Windows XP, MSVS 7.1
  SET(GSL_POSSIBLE_ROOT_DIRS
    ${GSL_ROOT_DIR}
    $ENV{GSL_ROOT_DIR}
    ${GSL_DIR}
    ${GSL_HOME}    
    $ENV{GSL_DIR}
    $ENV{GSL_HOME}
    $ENV{EXTRA}
    "C:/Programs/gsl/"
    "C:/Program Files (x86)/gsl/"
    "C:/Program Files/gsl/"
    "C:/Programs/"
    "C:/Program Files/"
    )
  IF(GSL_USE_STATIC_LIBS)
    SET(GSLSTATIC lib)
  ENDIF(GSL_USE_STATIC_LIBS)  
	 

  FIND_PATH(GSL_INCLUDE_DIR
    NAMES gsl/gsl_cdf.h gsl/gsl_randist.h
    HINTS ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES include
    DOC "GSL header include dir"
    )
  
  FIND_LIBRARY(GSL_GSL_LIBRARY
    NAMES "${GSLSTATIC}gsl"
    HINTS  ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES lib
    DOC "GSL library dir" )  
  
  FIND_LIBRARY(GSL_GSLCBLAS_LIBRARY
    NAMES "${GSLSTATIC}gslcblas" "${GSLSTATIC}cblas"
    HINTS  ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES lib
    DOC "GSL cblas library dir" )
  
  SET(GSL_LIBRARIES ${GSL_GSL_LIBRARY})

  MARK_AS_ADVANCED(
	GSL_GSLCBLAS_LIBRARY
	GSL_GSL_LIBRARY
	GSL_INCLUDE_DIR
  )


ELSE(WIN32)
  
  IF(UNIX) 
    SET(GSL_CONFIG_PREFER_PATH 
      "$ENV{GSL_DIR}/bin"
      "$ENV{GSL_DIR}"
      "$ENV{GSL_HOME}/bin" 
      "$ENV{GSL_HOME}" 
      CACHE STRING "preferred path to GSL (gsl-config)")
    FIND_PROGRAM(GSL_CONFIG gsl-config
      HINTS ${GSL_CONFIG_PREFER_PATH}
      /usr/bin/
      DOC "Path to gsl-config binary"
      )
    # MESSAGE("DBG GSL_CONFIG ${GSL_CONFIG}")
    
    IF (GSL_CONFIG) 
      # set CFLAGS to be fed into C_FLAGS by the user:
      EXEC_PROGRAM(${GSL_CONFIG}
        ARGS --cflags
        OUTPUT_VARIABLE GSL_CONFIG_CFLAGS )      
      SET(GSL_C_FLAGS ${GSL_CONFIG_CFLAGS}
      	CACHE STRING "Flags used to compile against GSL")
	
	IF (NOT GSL_PREFIX)      
      EXEC_PROGRAM(${GSL_CONFIG}
        ARGS --prefix
        OUTPUT_VARIABLE GSL_PREFIX)
      SET(GSL_PREFIX ${GSL_PREFIX} CACHE STRING "Location of GSL")
      MESSAGE(STATUS "Using GSL from ${GSL_PREFIX}")
    ENDIF(NOT GSL_PREFIX)
    
      # set INCLUDE_DIRS to prefix+include
      SET(GSL_INCLUDE_DIR ${GSL_PREFIX}/include
      	CACHE STRING "Location of GSL headers")
     
      
      # set link libraries and link flags
      EXEC_PROGRAM(${GSL_CONFIG}
        ARGS --libs
        OUTPUT_VARIABLE GSL_CONFIG_LIBS )
      STRING(REGEX MATCHALL "[-][l][^ ;]+" 
        GSL_LIBRARIES_WITH_PREFIX 
        "${GSL_CONFIG_LIBS}" )
      IF(GSL_USE_STATIC_LIBS)
      	IF(UNIX)
      		STRING(REGEX REPLACE "[-][l]([^ ;]+)" "lib\\1.a"  GSL_LIBRARIES_WITH_PREFIX  "${GSL_LIBRARIES_WITH_PREFIX}")
      	ENDIF(UNIX)
      ENDIF(GSL_USE_STATIC_LIBS)
      SET(GSL_LIBRARIES ${GSL_LIBRARIES_WITH_PREFIX}
      	CACHE STRING "GSL libraries to link against")

      # extract link dirs for rpath  
      # split off the link dirs (for rpath)
      # use regular expression to match wildcard equivalent "-L*<endchar>"
      # with <endchar> is a space or a semicolon
      STRING(REGEX MATCHALL "[-][L]([^ ;])+" 
        GSL_LINK_DIRECTORIES_WITH_PREFIX 
        "${GSL_CONFIG_LIBS}" )
      #      MESSAGE("DBG  GSL_LINK_DIRECTORIES_WITH_PREFIX=${GSL_LINK_DIRECTORIES_WITH_PREFIX}")

      # remove prefix -L because we need the pure directory for LINK_DIRECTORIES
      
      IF (GSL_LINK_DIRECTORIES_WITH_PREFIX)
        STRING(REGEX REPLACE "[-][L]" "" GSL_LINK_DIRECTORIES ${GSL_LINK_DIRECTORIES_WITH_PREFIX} )
      ENDIF (GSL_LINK_DIRECTORIES_WITH_PREFIX)
      SET(GSL_EXE_LINKER_FLAGS "-Wl,-rpath,${GSL_LINK_DIRECTORIES}"
      	CACHE STRING "Flags used when linking against GSL")
      #      MESSAGE("DBG  GSL_LINK_DIRECTORIES=${GSL_LINK_DIRECTORIES}")
      #      MESSAGE("DBG  GSL_EXE_LINKER_FLAGS=${GSL_EXE_LINKER_FLAGS}")

      #      ADD_DEFINITIONS("-DHAVE_GSL")
      #      SET(GSL_DEFINITIONS "-DHAVE_GSL")
      MARK_AS_ADVANCED(
        GSL_C_FLAGS
        GSL_INCLUDE_DIR
        GSL_LIBRARIES
        GSL_LINK_DIRECTORIES
        GSL_DEFINITIONS
        GSL_EXE_LINKER_FLAGS
        GSL_CONFIG
        GSL_PREFIX
        GSL_CONFIG_PREFER_PATH
	)
      
    ELSE(GSL_CONFIG)
      MESSAGE("FindGSL.cmake: gsl-config not found. Please set it manually. GSL_CONFIG=${GSL_CONFIG}")
    ENDIF(GSL_CONFIG)

  ENDIF(UNIX)
ENDIF(WIN32)


IF(GSL_LIBRARIES)
  IF(GSL_INCLUDE_DIR OR GSL_C_FLAGS)

    SET(GSL_FOUND 1)
    
  ENDIF(GSL_INCLUDE_DIR OR GSL_C_FLAGS)
ENDIF(GSL_LIBRARIES)

