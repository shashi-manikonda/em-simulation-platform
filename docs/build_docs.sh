#!/bin/bash
# A script to automate the Sphinx documentation build process.

# Clean the previous build
make -C /app/docs clean

# Build the HTML documentation
make -C /app/docs html