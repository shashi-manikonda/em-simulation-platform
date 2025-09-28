#!/bin/bash
# A script to automate the Sphinx documentation build process.

# Clean the previous build
make -C docs clean

# Build the HTML documentation
make -C docs html
