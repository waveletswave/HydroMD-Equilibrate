# HydroMD-Equilibrate: Equilibration Analysis in Molecular Dynamics Simulations

## Introduction
HydroMD-Equilibrate is a Python-based analytical tool designed to detect equilibrium states in molecular dynamics simulations. Developed during a class exercise, this tool epitomizes the intersection of statistical analysis and physical simulation, reflecting a commitment to interdisciplinary learning and application.

## Background
Determining when a molecular system has reached equilibrium is a pivotal step in simulation analysis. By integrating the Mann-Kendall trend test and Sen's slope estimator, HydroMD-Equilibrate offers a novel statistical approach to this challenge, broadening the scope of traditional methods used in the field of molecular dynamics.

## Features
- Performs time series analysis to detect equilibration in molecular dynamics simulations.
- Employs the Mann-Kendall test to identify trends within the data.
- Uses Sen's slope estimator to quantify the rate of change and assess stability.
- Pinpoints equilibration points, enhancing the interpretability of simulation results.

## Requirements
- Python 3.x
- NumPy
- pymannkendall
- matplotlib

## Installation
Clone the repository to your local machine for easy access and updates:
```bash
git clone https://github.com/waveletswave/HydroMD-Equilibrate.git
```

## Usage
Execute the HydroMD-Equilibrate script to process your data:
```bash
python HydroMD-Equilibrate.py
```
The script will output a plot illustrating the temperature trends, equilibration points, and Sen's slope, providing a visual cue for equilibrium detection.

### Visualization
Below is an example output from HydroMD-Equilibrate, showcasing temperature data versus time, with equilibration points and Sen's slope highlighted:

![Temperature vs. Time with Equilibration Points and Sen's Slope](/Users/benthosyy/Downloads/HydroMD-Equilibrate.png)

## Contributing
Contributions are welcome! Whether you are a seasoned researcher or a student exploring molecular dynamics, your insights can help enhance HydroMD-Equilibrate. We value a culture of collaboration and open dialogue across disciplines.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
This tool is a product of academic curiosity and interdisciplinary dialogue. Special thanks to the academic community for fostering an environment where such cross-pollination of ideas is encouraged.
