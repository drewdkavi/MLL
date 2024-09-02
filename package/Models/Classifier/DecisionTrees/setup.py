from distutils.core import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            Extension('generateRule', ['generateRule.pyx']),
        ],
        compiler_directives={"language_level": "3", "profile": True}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)