from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            "package//Models//Neural_Network//SNN.pyx",
            "package//Models//Neural_Network//SNN2.pyx"
        ],
        compiler_directives={"language_level": "3", "profile": True}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)