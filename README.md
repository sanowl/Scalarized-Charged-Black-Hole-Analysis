# Scalarized Charged Black Hole Analysis

This project implements numerical methods to analyze the stability and properties of scalarized charged black holes in Einstein-Maxwell theory minimally coupled with a non-linear complex scalar field. The analysis is based on the research paper "Stability analysis on charged black hole with non-linear complex scalar" by Zhan-Feng Mai and Run-Qiu Yang.

## Overview

The code provides tools to:

1. Solve the equations of motion for scalarized charged black holes
2. Analyze thermodynamic stability in different ensembles
3. Compute quasi-normal modes using the WKB approximation method
4. Visualize results and compare with Reissner-Nordström (RN) black holes

## Key Features

- `ScalarizedChargedBlackHole` class for encapsulating black hole properties and methods
- Numerical integration of coupled differential equations
- Shooting method to find scalarized solutions with target mass and charge
- Computation of thermodynamic quantities (mass, charge, temperature, chemical potential)
- Analysis of stability in microcanonical, canonical, and grand canonical ensembles
- WKB approximation for quasi-normal modes calculation
- Visualization of metric functions, matter fields, and thermodynamic properties

## Main Findings

The code can be used to reproduce and extend the key findings from the paper:

1. Scalarization cannot result from a continuous phase transition for general scalar potential
2. Possible first-order phase transitions from RN to scalarized black holes in microcanonical and canonical ensembles
3. Stability analysis using quasi-normal modes
4. Counterexample to the Penrose-Gibbons conjecture
5. Proposal of new versions of Penrose inequality in the charged case

## Usage

To use the code:

1. Adjust the parameters in the `ScalarizedChargedBlackHole` initialization
2. Run the main analysis section to solve for scalarized solutions
3. Analyze stability and compute quasi-normal modes
4. Visualize results using the provided plotting functions

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## References

Mai, Z. F., & Yang, R. Q. (2021). Stability analysis on charged black hole with non-linear complex scalar. arXiv:2101.00026v4 [gr-qc]

## Note

This code is for research and educational purposes. The results should be carefully verified and compared with analytical predictions where possible.