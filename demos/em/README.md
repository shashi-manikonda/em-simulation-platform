# EM Demo Scripts

This directory contains scripts that demonstrate the capabilities of the `em-app` package for electromagnetic field simulation and analysis.

## Running the Demos

To run all the demos in the recommended order, execute the following command from the root of the project:

```bash
python run_all_demos.py
```

The output of the scripts will be saved in the `runoutput/em` directory.

## Demo Scripts

The demos are numbered to be run in a sequence that builds from basic validation to more complex examples.

1.  **`01_validation_demo.ipynb`**: This script validates the accuracy of the Biot-Savart solver by comparing its output to known analytical solutions for simple geometries.
2.  **`02_dipole_approximation_demo.ipynb`**: This demo showcases the dipole approximation for a current loop and compares it to the full Biot-Savart calculation.
3.  **`03_plotting_capabilities_demo.ipynb`**: This script demonstrates the various plotting and visualization features of the `em-app` package.
4.  **`04_helmholtz_coil_demo.py`**: This script calculates and visualizes the magnetic field of a Helmholtz coil, a device used to produce a region of nearly uniform magnetic field.