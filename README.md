# SensitivityAnalysis: 

## Introduction
Sensitivity Analysis is an adaptive numerical differentiation tool that improves the accuracy of computing the scaled and unscaled derivatives (Sensitivities/Control Coefficients/Elasticity Coefficients) of models built using Tellurium. The algorithm is based off of the richardson extrapolation method. These sensitivities are of interest in terms of modeling the effects of perturbations on steady state. 

## Getting Started
### Installing Directly from Github
1. Download SensitivityAnalysis. Click the green button at the top of this page that says "Code" and select "Download ZIP". Unzip the file into the desired working directory
2. Navigate to the working directory
3. Install the base dependencies. Run `pip install -r requirements.txt`.
   - tellurium
   - numpy
4. Finally test the package using `python SensitivityAnalysisTest.py`. Once the setup is complete you will only need to open the terminal, navigate into the folder that contains the package.


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
   1. getCC: Returns the scaled (flux/concentration) control coefficient with respect to a global parameter
      - variable: reaction or species concentration
      - parameter: kinetic constant or boundary species
   3. getuCC: Returns the unscaled (flux/concentration) control coefficient with respect to a global parameter
      - variable: reaction or species concentration
      - parameter: kinetic constant or boundary species
   5. getEE: Returns the scaled elasticity coefficient with respect to a global parameter
      - variable: a reaction (e.g. 'J1')
      - parameter: the independent parameter, for example a kinetic constant, floating or boundary species
   7. getuEE: Returns the unscaled elasticity coefficient with respect to a global parameter
      - variable: a reaction (e.g. 'J1')
      - parameter: the independent parameter, for example a kinetic constant, floating or boundary species

### Documentation
For the full documentation see: https://sensitivityanalysis.readthedocs.io/

### Terminology
1. Control Coefficient: Describe how much influence a given reaction step has on a steady state flux or species concentration level
   1. Flux Control Coefficient: The fractionl change in flux brought a out by a given fractional change in enzyme (protein) concentration
   2. Concentration Control Coefficient: The fractional change in species concentration given a fractional change in enzyme (protein) concentration
2. Flux: steady state rate through a pathway
3. Elasticity Coefficient: Also known as the kinetic order, or the derivative of the reaction rate with respect to the species concentration.
   - The fractional change in reaction rate in response to a fractional change in a given reactant or product while keeping all other reactants and products constant. 
   - For a species that increases reaction rate: Elasticity is positive
   - For a species that decreases reaction rate: Elasticity is negative
5. Scaled: Unitless
6. Unscaled: Contains units
