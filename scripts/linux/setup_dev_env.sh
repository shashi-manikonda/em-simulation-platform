#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Default Compiler
COMPILER="ifx"

# Parse Arguments
for arg in "$@"; do
    case $arg in
        --compiler=*)
        COMPILER="${arg#*=}"
        shift
        ;;
        -c)
        COMPILER="$2"
        shift 2
        ;;
    esac
done

# Validate Compiler
if [[ "$COMPILER" != "ifx" && "$COMPILER" != "gfortran" ]]; then
    echo "Error: Invalid compiler '$COMPILER'. Use 'ifx' or 'gfortran'."
    exit 1
fi

echo ":: Selected COSY Compiler: $COMPILER"

# 1. Initialize Intel OneAPI Environment (Only needed for ifx, but good to have environment generally)
if [ "$COMPILER" == "ifx" ]; then
    echo ":: Initializing Intel OneAPI environment..."
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

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

# 2.5 Update COSY Configuration
CONFIG_FILE="$SANDALWOOD_DIR/src/sandalwood/backends/cosy/cosy_config.env"
if [ -f "$CONFIG_FILE" ]; then
    echo ":: Updating config to use $COMPILER..."
    # Update the export line
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/export COSY_COMPILER=.*/export COSY_COMPILER=$COMPILER/" "$CONFIG_FILE"
    else
        sed -i "s/export COSY_COMPILER=.*/export COSY_COMPILER=$COMPILER/" "$CONFIG_FILE"
    fi
else
    echo "Warning: Config file not found at $CONFIG_FILE"
fi

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
