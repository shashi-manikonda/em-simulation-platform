import sys
from setuptools import setup, find_packages, Extension
import numpy
import pybind11

# Set compiler arguments based on the operating system
if sys.platform == 'win32':
    # MSVC compiler arguments
    cpp_args = ['/std:c++17', '/openmp']
    link_args = []
else:
    # GCC/Clang compiler arguments
    cpp_args = ['-std=c++17', '-fopenmp', '-O3', '-march=native']
    link_args = ['-fopenmp']

extensions = [
    Extension('mtflib.backends.cpp.mtf_cpp',
              ['src/mtflib/backends/cpp/mtf_data.cpp', 'src/mtflib/backends/cpp/pybind_wrapper.cpp', 'src/mtflib/backends/cpp/biot_savart_ops.cpp'],
              include_dirs=[pybind11.get_include(), numpy.get_include(), 'src/mtflib/backends/cpp'],
              language='c++',
              extra_compile_args=cpp_args,
              extra_link_args=link_args),
    Extension('mtflib.backends.c.mtf_c_backend',
              ['src/mtflib/backends/c/c_backend.cpp', 'src/mtflib/backends/c/c_pybind_wrapper.cpp'],
              include_dirs=[pybind11.get_include(), numpy.get_include(), 'src/mtflib/backends/c'],
              language='c++',
              extra_compile_args=cpp_args,
              extra_link_args=link_args),
]

setup(
    ext_modules=extensions,
    zip_safe=False,
)