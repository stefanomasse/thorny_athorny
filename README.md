This code and data allow to reproduce the computational modeling part of the manuscript:

# Sub-type Specific Connectivity between CA3 Pyramidal Neurons May Underlie Their Sequential Activation during Sharp Waves

**Authors:** Rosanna P. Sammons*, Stefano Masserini*, Laura Moreno-Velasquez, Verjinia D. Metodieva, Gaspar Cano, Andrea Sannio, Marta Orlando, Nikolaus Maier, Richard Kempter, Dietmar Schmitz

**Publication:** eLife13:RP98653

**DOI**: https://doi.org/10.7554/eLife.98653.2.sa2

---

## Content

In this repository, you can find the following contents:

### network_code

This is a folder, containing the necessary python code to set up and simulate the spiking model, namely:

- `utils.py`: contains helper functions used to analyze the outcome of network simulations
- `parameters.py`: contains the default model parameters, as well as the default values of the varialbes used to analyze the simulations
- `network.py`: sets up the spiking network with 4 populations in the Brian2 library
- `simulations.py`: runs the spiking network and analyzes relevant variables, especially to quantify features of SPW events.

### Main files

The main folder contains python files and notebooks that use the code in the the `network_code/` folder in order to run simulations and produce the figures in the paper:
 
  - `functions_for_parameters_sweeps.py`: contains the functions used to simulate the network with different connectivity values and to analyze the outcomes. Namely
      these functions allow to first find the EE synaptic weights to balance network activity, then to simulate the balanced network and save the prepocessed results,
      then to further analyze the results and prepare them for plotting. These functions come in versions for either 1 or 2 parameters.
  - `parameter_sweeps.py`: sequentially uses the `functions_for_parameters_sweeps.py` to investigate the effects of varying 1 or 2 connectivities in different model configurations.
      The outcomes are saved in the `data/` folder.
  - `simulate_network.ipynb`: this Jupyter notebook simulates the spiking network in three different "default" configurations and plots the sample SPWs in Figures 4, 4 supplement 2, and 4 supplement 3.
      The traces are saved in the `samples/network_traces` folder.
  - `plt_all_singles.ipynb`: this Jupyter notebook loads the data in the `data/` folder and plots the 1-dimensional parameter sweeps in Figures 4, 4 supplement 2, and 4 supplement 3.
  - `plt_all_couples.ipynb`: this Jupyter notebook loads the data in the `data/` folder and plots the 2-dimensional parameter sweeps in Figure 5
  - `generate_fi_curves.ipynb`: this Jupyter notebook implement simple functions to simulate single AdEx neurons and then uses them to plot the f-I curves in Figure 4, supplement 1.
      The traces are saved in the `samples/single_neuron_traces` folder.

### data

This is a folder containing the most important intermediate and final data produced by parameter sweeps:

- `exc_cond.npy` files contain the EE synapitc weights allowing to balance the SPW size, for each combination of varied connectivities and model configurations.
    in `functions_for_parameters_sweeps`. Each of them is an array with entries corresponding to a different parameter(s) value. Using these pre-computed files allows to avoid
    computationally expensive runs of the `find_balance` functions.
- `data.pkl` files contain the most important variables obtained after simulating and analyzing the model for each combination of varied connectivities and model configurations.
    These data come as a dictionary, whose keys contain  array with entries corresponding to a different parameter(s) value. Using these pre-computed files allows to avoid
    running the `simulate` functions in `functions_for_parameters_sweeps` and to directly plot the outcomes in Figures 4, 5, 4 supplement 2, and 4 supplement 3.
  
### samples

- `single_neuron_traces/`: folder containing pre-computed time traces of relevant variables, used to plot the single neuron f-I curves in Figure 4, supplement 1.
- `network_traces/`: folder containing pre-computed time traces of relevant variables, used to plot the network samples in Figures 4, 4 supplement 2, and 4 supplement 3.

## How to use the scripts

### Generate Manuscript Figures

All the computational figures in the manuscript can be generated in a timely manner by running the Jupyter notebooks `simulate_network.ipynb`, `plt_all_singles.ipynb`, `plt_all_couples.ipynb`,
and `generate_fi_curves.ipynb`. These notebooks will exploit the pre-computed data in the `data/` and `samples/` folders.

### Reproduce Manuscript Pipeline

All the steps needed to generate the data can be reproduced by running `parameter_sweeps.py`. Note that this file exploits multicore computing and it is meant to be run on a high-performance
computing cluster. This file can be easily customized to run just a subset of the parameter sweeps, or to run them for a shorter time. It is also possible to skip the computationally expensive
balancing step by using the pre-computed `exc_cond.npy` files in the `data\` folder.

## Contact

For any questions, the author of the code is available at stefanomasse@gmail.com
