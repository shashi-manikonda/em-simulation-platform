#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Initialize Intel OneAPI Environment
echo ":: Initializing Intel OneAPI environment..."
source /opt/intel/oneapi/setvars.sh || true

# 2. Define Absolute Paths
CURRENT_DIR=$(pwd)
# Resolve absolute path for sandalwood
if [ -d "../sandalwood" ]; then
    SANDALWOOD_DIR=$(cd ../sandalwood && pwd)
else
    echo "Error: Sandalwood directory not found at ../sandalwood"
    exit 1
fi

echo ":: Setup Context:"
echo "   Em-App Dir: $CURRENT_DIR"
echo "   Sandalwood Dir: $SANDALWOOD_DIR"

# 3. Build COSY Backend (Fortran)
echo ":: Building COSY Backend..."
cd "$SANDALWOOD_DIR/src/sandalwood/backends/cosy"
# Ensure the compile script is executable
chmod +x ./compile_cosy.sh
./compile_cosy.sh

# 4. Install Sandalwood (Editable Mode)
echo ":: Installing Sandalwood..."
cd "$SANDALWOOD_DIR"
# Install with 'uv' - this assumes 'uv' is in PATH.
uv pip install -e .

# 5. Install Em-App (Editable Mode with Dev Dependencies)
echo ":: Installing Em-App..."
cd "$CURRENT_DIR"
uv pip install -e ".[dev]"

echo ":: Setup Complete! Environment is ready."
echo ":: To run tests: pytest"
