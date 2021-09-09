# SensitivityAnalysis
Code for Sensitivity Analysis, computing the control coefficients and elasticity coefficients

## Introduction
Sensitivity Analysis is a numerical differentiation tool that improves the accuracy of computing the scaled and unscaled derivatives (Sensitivities/Control Coefficients/Elasticity Coefficients) of models built using Tellurium.

## Getting Started
### Installing with Pypi


### Installing Directly from 


### Usage
Steps to using:
1. Import Tellurium and SensitivityAnalysis
   ```
   import tellurium as te
   import SensitivityAnalysis as sa
   ```
3. Create a tellurium model
   - Show example model
4. Identify parameters and variables of interest
5. Call one of the built in SensitivityAnalysis methods
   1. getCC:
   2. getuCC:
   3. getEE:
   4. getuEE:

## Testing
To test the functionality of the SensitivityAnalysis package:
1. Ensure that SensitivityAnalysis.py and SensitivityAnalysisTest.py are both in the current working directory
2. run the following command from your terminal ```python SensitivityAnalysisTest.py```
   - Ensure that the output says that all tests have been passed

## Contributing
