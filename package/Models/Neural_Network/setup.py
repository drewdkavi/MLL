from distutils.core import setup
from setuptools import Extension
import os
from Cython.Build import cythonize
import numpy


extensions = [
    Extension('SNN', ['SNN.pyx']),
    Extension('SNN2', ['SNN2.pyx'])
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "profile": True}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)