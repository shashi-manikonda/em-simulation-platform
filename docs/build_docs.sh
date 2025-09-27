#!/bin/bash
# A script to automate the Sphinx documentation build process.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Clean the previous build
make -C "${SCRIPT_DIR}" clean

# Build the HTML documentation
make -C "${SCRIPT_DIR}" html