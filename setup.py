from setuptools import setup, Extension
import os
os.environ["CC"] = "/opt/homebrew/bin/gcc-13"

# Define the extension module
extensions = [Extension('random_generator', ['random_generator.c'])]

# Add OpenMP and optimization flags for GCC or Clang (Linux and macOS)
if os.name == 'posix':
    extensions[0].extra_compile_args = ['-fopenmp', '-O3']  # Add optimization flags here
    extensions[0].extra_link_args = ['-fopenmp']

setup(
    name='random_generator',
    version='1.0',
    ext_modules=extensions,
)