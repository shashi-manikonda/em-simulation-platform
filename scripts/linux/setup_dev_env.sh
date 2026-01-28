#!/bin/bash
# Function to handle exit/return behavior
die() {
    echo "Error: $1"
    # Check if the script is being sourced
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        return 1
    else
        exit 1
    fi
}

setup_dev_env() {
    # Default Compiler
    local COMPILER="ifx"

    # Parse Arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --compiler=*)
            COMPILER="${1#*=}"
            shift
            ;;
            -c)
            COMPILER="$2"
            shift 2
            ;;
            *)
            shift # Ignore unknown args or handle them?
            ;;
        esac
    done

    # Validate Compiler
    if [[ "$COMPILER" != "ifx" && "$COMPILER" != "gfortran" ]]; then
        die "Invalid compiler '$COMPILER'. Use 'ifx' or 'gfortran'."
        return 1 2>/dev/null || exit 1
    fi

    echo ":: Selected COSY Compiler: $COMPILER"

    # 1. Initialize Intel OneAPI Environment (Only needed for ifx)
    if [ "$COMPILER" == "ifx" ]; then
        echo ":: Initializing Intel OneAPI environment..."
        # Use a subshell check or similar to avoid exiting if not found? 
        # Actually setvars.sh might set env vars, so we need to source it.
        # We allow it to fail without crashing.
        if [ -f /opt/intel/oneapi/setvars.sh ]; then
             source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || echo "Warning: Failed to source setvars.sh"
        else
             echo "Warning: /opt/intel/oneapi/setvars.sh not found."
        fi
    fi

    # 2. Define Absolute Paths
    local CURRENT_DIR=$(pwd)
    local SANDALWOOD_DIR
    
    # Resolve absolute path for sandalwood (assumes sibling directory structure)
    if [ -d "../sandalwood" ]; then
        SANDALWOOD_DIR=$(cd ../sandalwood && pwd)
    else
        die "Sandalwood directory not found at ../sandalwood. Please ensure sandalwood is checked out as a sibling to em-simulation-platform."
        return 1 2>/dev/null || exit 1
    fi

    echo ":: Setup Context:"
    echo "   Em-App Dir: $CURRENT_DIR"
    echo "   Sandalwood Dir: $SANDALWOOD_DIR"

    # 2.5 Update COSY Configuration
    local CONFIG_FILE="$SANDALWOOD_DIR/src/sandalwood/backends/cosy/cosy_config.env"
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
    # We must not cd in the main shell if sourced, or we change the user's PWD.
    # Use subshell or pushd/popd if we want to return to original dir, 
    # BUT if we fail, we might stay there.
    # Safe way: use subshells for execution steps where env vars don't need to persist.
    
    (
        cd "$SANDALWOOD_DIR/src/sandalwood/backends/cosy" || exit 1
        chmod +x ./compile_cosy.sh
        ./compile_cosy.sh
    )
    if [ $? -ne 0 ]; then
        die "COSY Backend compilation failed."
        return 1 2>/dev/null || exit 1
    fi

    # 4. Install Sandalwood (Editable Mode)
    echo ":: Installing Sandalwood..."
    (
        cd "$SANDALWOOD_DIR" || exit 1
        uv pip install -e .
    )
     if [ $? -ne 0 ]; then
        die "Sandalwood installation failed."
        return 1 2>/dev/null || exit 1
    fi

    # 5. Install Em-App (Editable Mode with Dev Dependencies)
    echo ":: Installing Em-App..."
    # No need to cd if we are already in CURRENT_DIR (which we assume is where script is run, or we should use CURRENT_DIR)
    # The original script assumed usage from root of repo? 
    # "CURRENT_DIR=$(pwd)" suggests it trusts pwd.
    
    (
        cd "$CURRENT_DIR" || exit 1
        uv pip install -e ".[dev]"
        pre-commit install
    )
    if [ $? -ne 0 ]; then
        die "Em-App installation failed."
        return 1 2>/dev/null || exit 1
    fi

    echo ":: Setup Complete! Environment is ready."
    echo ":: To run tests: pytest"
}

# Run the function, passing all arguments
setup_dev_env "$@"
