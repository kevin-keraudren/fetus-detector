from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("_image_features", ["lib/_image_features.pyx"],
                           language="c++",
                           include_dirs=get_numpy_include_dirs()+['lib',"/vol/medic02/users/kpk09/LOCAL/include"],
                           library_dirs=["/vol/medic02/users/kpk09/LOCAL/lib"],
                   libraries=["tbb"]),
             ])

